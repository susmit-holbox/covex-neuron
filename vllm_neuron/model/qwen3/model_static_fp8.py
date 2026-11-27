# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Static FP8 Implementation
===============================

Static-FP8 variant of :mod:`vllm_neuron.model.qwen3.model` for TRN2+.

Weights are stored in ``float8_e4m3fn`` with a single per-tensor dequant
scale. Activations are static-FP8 quantized for QKV/O proj and MLP. Both
weight and input scales are pre-broadcast to ``(128, 1)`` fp32 at load
time (or ``(128, 3)`` for fused QKV) to avoid a runtime broadcast.

Adapted from :mod:`vllm_neuron.model.llama3.model_static_fp8`, with these
Qwen3-specific changes:

  * Per-head QK-norm is applied before RoPE on both prefill (via
    ``NF.qkv_proj``'s ``qk_norm_pre_rope_*`` kwargs) and decode (via
    ``NF.attention_decode``'s ``rmsnorm_QK_pre_rope_*`` kwargs). Without
    it the model is numerically wrong even though shapes match.
  * The Qwen3 bf16 forward skeleton is used: Qwen3's external KV scatter,
    ``build_per_head_block_table`` + ``scatter_new_kv`` for GQA decode,
    and no input-norm fusion into the decode megakernel.
  * Qwen3 has no DCP, attention-DP, draft-MLP groups, packed-KV fallback
    or fused-decode-mask machinery. None of it is reproduced here.
  * The MLP still owns ``post_attention_layernorm`` (the Llama3 fused-norm
    contract): the decoder layer skips the external norm and hands the
    weight plus epsilon to :class:`Qwen3MLPStaticFp8`.

This implementation targets checkpoints quantized with
``quant_method='modelopt'`` and ``quant_algo='FP8'``. ``lm_head`` is not
quantized (ModelOpt excludes it) and stays BF16.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import NormType, QuantizationType
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
)

from ..kv_cache import paged_slot_indices
from ..qwen3_vl.utils.decode_kv import build_per_head_block_table, scatter_new_kv
from .config import Qwen3Config
from .model import (
    Qwen3Attention as Qwen3AttentionBf16,
    Qwen3DecoderLayer as Qwen3DecoderLayerBf16,
    Qwen3ForCausalLM as Qwen3ForCausalLMBf16,
    Qwen3MLP as Qwen3MLPBf16,
    Qwen3Model as Qwen3ModelBf16,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from .weight_loaders_static_fp8 import (
    _FP8_DTYPE,
    _PMAX,
    _QKV_FUSED,
    attach_attention_loaders,
    attach_mlp_loaders,
)


# =============================================================================
# Attention (static FP8)
# =============================================================================


class Qwen3AttentionStaticFp8(Qwen3AttentionBf16):
    """GQA attention with TP head sharding, QK-norm and static FP8 weights.

    Structurally identical to :class:`Qwen3AttentionBf16` except:
      * QKV / O weights are allocated in ``float8_e4m3fn``.
      * Four scale buffers are registered (qkv weight/input, o weight/input).
      * Prefill QKV/O projections pass STATIC kernel kwargs and the
        per-head QK-norm gamma.
      * Decode passes STATIC kwargs plus ``rmsnorm_QK_pre_rope_*``.

    Qwen3 bf16 intentionally does not fuse the attention input RMSNorm into
    the decode megakernel (the fusion regressed Qwen3-32B BF16 LongBench;
    see ``model.py``'s decoder-layer comment), so the decoder layer still
    runs ``input_layernorm`` externally and this module leaves
    ``rmsnorm_X_enabled=False``.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)

        # Replace the bf16 QKV/O weights with static-FP8 storage. The shapes
        # are the same as the bf16 sibling; only the dtype changes.
        del self.qkv_proj_weight
        del self.o_proj_weight

        qkv_size = self.q_size + 2 * self.kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim // self.world_size
        )

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=_FP8_DTYPE)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=_FP8_DTYPE)
        )

        # Static dequant / input scales. Shapes match the kernel contract
        # ([_PMAX, 3] for fused QKV weight, [_PMAX, 1] elsewhere).
        self.register_buffer(
            "qkv_weight_scale",
            torch.empty(_PMAX, _QKV_FUSED, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "qkv_input_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_weight_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_input_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )

        # KV cache quantization scales (populated at weight-load time by
        # Qwen3ForCausalLMStaticFp8._load_kv_cache_scales, inherited).
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    # ------------------------------------------------------------------
    # Weight loaders
    # ------------------------------------------------------------------

    def _setup_weight_loaders(self) -> None:
        """Attach static-FP8 loaders to the fp8 weights and scale buffers.

        The bf16 base ``__init__`` calls this before the fp8 parameters and
        scale buffers exist, so the guard forwards that first call to the
        bf16 loader. The second call (after the parameters are replaced)
        attaches the static loaders.
        """
        if not hasattr(self, "qkv_weight_scale"):
            return super()._setup_weight_loaders()

        attach_attention_loaders(
            self,
            q_size=self.q_size,
            kv_size=self.kv_size,
            world_size=self.world_size,
            num_kv_replicas=self.num_kv_replicas,
            # Qwen3 is dense: no attention DP and no KV all-to-all.
            attention_dp_size=1,
            attention_dp_rank=0,
            kv_needs_a2a=False,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
        )

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to prefill or decode (Qwen3 bf16 structure)."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # The decoder layer applies input_layernorm externally on both paths,
        # so this gather is the only SP boundary before the QKV projection.
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill path with STATIC QKV/O projections and pre-RoPE QK-norm."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        # STATIC kernels take a bf16 activation and quantize internally.
        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)

        # Step 1: fused QKV projection + pre-RoPE QK-norm + RoPE.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            quantization_type=QuantizationType.STATIC,
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=self.qkv_input_scale,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            # Qwen3 per-head QK-norm runs BEFORE RoPE.
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
            qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, self.head_dim),
            qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, self.head_dim),
        ).squeeze(0)

        # The STATIC kernel emits the compute dtype; normalize defensively.
        qkv = qkv.to(self.dtype)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 3: external KV cache write (Qwen3 bf16 contract).
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        # FP8 KV cache quantize-on-write (kv_cache_dtype=fp8 only).
        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.v_cache.dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # Step 4: attention. Qwen3 has no sinks, no sliding window.
        if kv_segment_size:
            # The cache holds K*k_scale / V*v_scale (quantize-on-write above),
            # so fold the K dequant into the softmax scale and divide the V
            # dequant out of the output. Both are exact scalar no-ops at
            # k_scale == v_scale == 1 (bf16 KV). This is the static-FP8
            # reference's FP8-KV fold; the Qwen3 bf16 skeleton omits it.
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling / self.k_scale_float,
                tp_q=True,
                tp_out=True,
            )
            if self.v_scale_float != 1.0:
                attn_output = attn_output / self.v_scale_float
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # Step 5: output projection (STATIC).
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(
            attn_output,
            self.o_proj_weight,
            None,
            quantization_type=QuantizationType.STATIC,
            weight_scales=self.o_weight_scale,
            input_scales=self.o_input_scale,
        )
        attn_output = attn_output.squeeze(0).to(self.dtype)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        attention_mask: torch.Tensor | None = None,
    ):
        """Decode path (fused megakernel) with STATIC QKV/O and QK-norm."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"decode token count ({tokens}) must be a multiple of batch size "
            f"({B}); got S_decode={S_decode}"
        )

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        if attention_mask is None:
            assert positions is not None
            pos_ids = positions.view(1, B * S_decode)
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=pos_ids.to(torch.float32),
                bs=B,
                q_head=self.num_attention_heads_per_rank,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
            )

        # GQA (nkh > 1) needs the per-head 3D block table + 4D cache; a 2D
        # table routes GQA to the slow torch fallback.
        if nkh > 1:
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )

            output, K_new, V_new = NF.attention_decode(
                X=X,
                X_hidden_dim_actual=self.hidden_size,
                # Qwen3 bf16 deliberately keeps the input RMSNorm outside the
                # megakernel; the decoder layer applied it already.
                rmsnorm_X_enabled=False,
                W_qkv=self.qkv_proj_weight,
                bias_qkv=None,
                quantization_type_qkv=QuantizationType.STATIC,
                weight_dequant_scale_qkv=self.qkv_weight_scale,
                input_dequant_scale_qkv=self.qkv_input_scale,
                # Qwen3 per-head QK-norm runs BEFORE RoPE.
                rmsnorm_QK_pre_rope_enabled=True,
                rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
                rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, self.head_dim),
                rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, self.head_dim),
                rmsnorm_QK_post_rope_enabled=False,
                cos=cos_kernel,
                sin=sin_kernel,
                rope_contiguous_layout=True,
                K_cache_transposed=False,
                active_blocks_table=active_blocks_table,
                K_cache=self.k_cache,
                V_cache=self.v_cache,
                attention_mask=attention_mask,
                softmax_scale=self.scaling / self.k_scale_float,
                sink=None,
                update_cache=False,
                # STATIC fp8 weights: pass the raw fp8 O weight and scales;
                # the kernel applies v_scale. Never divide an fp8 weight.
                W_out=self.o_proj_weight,
                bias_out=None,
                quantization_type_out=QuantizationType.STATIC,
                weight_dequant_scale_out=self.o_weight_scale,
                input_dequant_scale_out=self.o_input_scale,
                transposed_out=False,
                out_in_sb=False,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                attention_dp=1,
                attention_dp_group=None,
                attention_dp_rank=0,
                kv_needs_a2a=False,
            )

            scatter_new_kv(
                self.k_cache,
                self.v_cache,
                K_new,
                V_new,
                slot_mapping,
                block_size,
                nkh,
                self.head_dim,
            )

            if self.world_size > 1:
                self.tp_group.all_reduce(output)

            return output.to(self.dtype)

        # nkh == 1 (TP >= num_kv_heads): the kernel takes the squeezed
        # [num_blocks, block_len, d_head] cache + 2D [B, num_blocks] table.
        k_cache = self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
        v_cache = self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache

        active_blocks_table = block_table.to(torch.int32)

        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.STATIC,
            weight_dequant_scale_qkv=self.qkv_weight_scale,
            input_dequant_scale_qkv=self.qkv_input_scale,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, self.head_dim),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, self.head_dim),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=False,
            W_out=self.o_proj_weight,
            bias_out=None,
            quantization_type_out=QuantizationType.STATIC,
            weight_dequant_scale_out=self.o_weight_scale,
            input_dequant_scale_out=self.o_input_scale,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        # External KV cache update (Qwen3 bf16 contract).
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        num_tokens = slot_mapping.shape[0]

        k_new = (
            K_new.permute(1, 2, 0)
            .reshape(B, nkh, S_decode, self.head_dim)
            .transpose(0, 1)
            .reshape(nkh, B * S_decode, self.head_dim)
        )
        k_new_flat = k_new.reshape(-1, self.head_dim)
        v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_new_flat.to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_new_flat.to(self.v_cache.dtype),
        )

        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output.to(self.dtype)


# =============================================================================
# MLP (static FP8)
# =============================================================================


class Qwen3MLPStaticFp8(Qwen3MLPBf16):
    """Fused pre-MLP RMSNorm + SwiGLU MLP with TP intermediate sharding.

    Unlike the bf16 sibling (which is MLP-only), this module owns **both**
    the post-attention RMSNorm and the MLP. Prefill runs
    :func:`NF.rmsnorm_quant` before :func:`NF.mlp` because the norm must
    run inside the sequence-parallel region; decode passes the norm to
    :func:`NF.mlp`, which places it. The decoder layer supplies ``ln_w``
    and ``eps`` and must not apply ``post_attention_layernorm`` itself.
    """

    def __init__(self, config: Qwen3Config):
        self._base_config = config
        super().__init__(config)

        # Replace the bf16 MLP weights with static-FP8 storage.
        del self.gate_proj_weight
        del self.up_proj_weight
        del self.down_proj_weight

        self.act_dtype = config.torch_dtype

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=_FP8_DTYPE,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=_FP8_DTYPE,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=_FP8_DTYPE,
            )
        )

        # Per-projection dequant scales + per-stage input scales. gate/up
        # share one input scale; down has its own.
        for name in (
            "gate_weight_scale",
            "up_weight_scale",
            "down_weight_scale",
            "gate_up_input_scale",
            "down_input_scale",
        ):
            self.register_buffer(
                name,
                torch.empty(_PMAX, 1, dtype=torch.float32),
                persistent=False,
            )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self, config: Qwen3Config | None = None) -> None:
        """Attach static-FP8 loaders to the MLP weights and scale buffers.

        The bf16 base ``__init__`` calls this before the fp8 parameters and
        scale buffers exist, so that first call forwards to the bf16 loader.
        """
        if not hasattr(self, "gate_weight_scale"):
            return super()._setup_weight_loaders(config or self._base_config)

        attach_mlp_loaders(
            self,
            intermediate_size_per_rank=self.intermediate_size_per_rank,
            mlp_tp_size=self.world_size,
            mlp_tp_rank=self.tp_group.rank_in_group,
            hidden_size=self.hidden_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        ln_w: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run the pre-MLP RMSNorm + static-FP8 MLP.

        * **Prefill:** explicit :func:`NF.rmsnorm_quant` before
          :func:`NF.mlp` (``norm_type=NO_NORM``). The norm must run inside
          the sequence-parallel region, before the all-gather.
        * **Decode:** :func:`NF.mlp` receives ``norm_type=RMS_NORM`` and
          the gamma; it fuses the norm or pre-quantizes internally.

        Either way the decoder layer must skip its plain
        ``post_attention_layernorm`` and pass the weight and epsilon here.
        """
        if ln_w is None:
            raise ValueError(
                "Qwen3MLPStaticFp8 requires ln_w on both prefill and decode; "
                "the decoder layer must pass "
                "post_attention_layernorm.weight."
            )

        if is_prefill:
            # Normalize and quantize on the local SP shard, then gather fp8.
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=QuantizationType.STATIC,
            )
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            mlp_norm_type = NormType.NO_NORM
            mlp_ln_w = None
        else:
            mlp_norm_type = NormType.RMS_NORM
            mlp_ln_w = ln_w.view(1, -1)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            eps=eps,
            ln_w=mlp_ln_w,
            norm_type=mlp_norm_type,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_weight_scale,
            up_w_scale=self.up_weight_scale,
            down_w_scale=self.down_weight_scale,
            gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale,
            output_dtype="bfloat16",
        )

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Decoder layer
# =============================================================================


class Qwen3DecoderLayerStaticFp8(Qwen3DecoderLayerBf16):
    """Single Qwen3 decoder layer with static-FP8 attention and MLP.

    The input RMSNorm stays a plain external op (Qwen3 bf16 avoids fusing
    it into the decode megakernel). The post-attention RMSNorm is owned by
    the MLP, so this layer skips it and forwards the weight plus epsilon.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.rms_norm_eps = config.rms_norm_eps

        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Qwen3AttentionStaticFp8(config, layer_idx=layer_idx)
        self.mlp = Qwen3MLPStaticFp8(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        # Keep the input norm external; Qwen3 bf16 does the same.
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        # The MLP owns post_attention_layernorm (Llama3 fused-norm contract):
        # do NOT normalize here, or the norm is applied twice.
        hidden_states = self.mlp(
            hidden_states,
            is_prefill=not is_decode,
            ln_w=self.post_attention_layernorm.weight,
            eps=self.rms_norm_eps,
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model backbone
# =============================================================================


class Qwen3ModelStaticFp8(Qwen3ModelBf16):
    """Qwen3 transformer backbone with static-FP8 layers.

    This bypasses ``Qwen3ModelBf16.__init__`` (as the MXFP8 sibling does)
    so it can wire the static decoder layers directly. All attributes the
    inherited bf16 forward reads are set here.
    """

    def __init__(self, config: Qwen3Config) -> None:
        nn.Module.__init__(self)
        self.config = config

        # Set by the runner via set_aux_hidden_state_layers(); the inherited
        # bf16 forward references it. Empty = no aux capture.
        self.aux_hidden_state_layers = []

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        # Static FP8 does not enable sequence-parallel decode.
        self.decode_sp_enabled = False

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayerStaticFp8(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )


# =============================================================================
# LM head + weight loading
# =============================================================================


class Qwen3ForCausalLMStaticFp8(Qwen3ForCausalLMBf16):
    """Qwen3 model with a BF16 LM head and static-FP8 linear layers.

    ModelOpt excludes ``lm_head`` from quantization, so it stays BF16.
    The constructor bypasses ``Qwen3ForCausalLMBf16.__init__`` (as the
    MXFP8 sibling does); the inherited bf16 ``forward`` and Eagle3 methods
    are reused.
    """

    def __init__(self, config: Qwen3Config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3ModelStaticFp8(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # <-- QUANTIZATION: lm_head is excluded from ModelOpt FP8, so BF16.
        # >>> PARALLELISM: column-parallel LM head (vocab sharded).
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    # ------------------------------------------------------------------
    # Static-FP8 scale sources
    # ------------------------------------------------------------------
    # Per-layer scale buffers and their checkpoint sources, relative to
    # ``model.layers.{i}``. The value lists feed each buffer's attached
    # loader: one slice for scalar -> [128, 1], three for fused QKV ->
    # [128, 3] (weight) or the Q/K/V input-scale consensus check.
    _FP8_ATTN_SCALE_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "qkv_weight_scale",
            (
                "self_attn.q_proj.weight_scale",
                "self_attn.k_proj.weight_scale",
                "self_attn.v_proj.weight_scale",
            ),
        ),
        (
            "qkv_input_scale",
            (
                "self_attn.q_proj.input_scale",
                "self_attn.k_proj.input_scale",
                "self_attn.v_proj.input_scale",
            ),
        ),
        ("o_weight_scale", ("self_attn.o_proj.weight_scale",)),
        ("o_input_scale", ("self_attn.o_proj.input_scale",)),
    )
    _FP8_MLP_SCALE_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("gate_weight_scale", ("mlp.gate_proj.weight_scale",)),
        ("up_weight_scale", ("mlp.up_proj.weight_scale",)),
        ("down_weight_scale", ("mlp.down_proj.weight_scale",)),
        ("gate_up_input_scale", ("mlp.gate_proj.input_scale",)),
        ("down_input_scale", ("mlp.down_proj.input_scale",)),
    )

    def _load_static_fp8_scales(
        self,
        checkpoint: SafetensorsCheckpoint,
        tp_rank: int,
        device: torch.device,
    ) -> None:
        """Load and assign each static-FP8 scale buffer on every layer.

        Calls the :class:`SafetensorsWeightLoader` attached to each scale
        buffer at FP8-module construction time, then :func:`setattr` the
        resulting tensor onto the owning sub-module. This replaces the
        meta-device allocation created under the model-runner's
        ``with torch.device("meta")`` context.

        ``load_sharded_pipelined`` only iterates ``named_parameters``; the
        scale buffers are ``persistent=False`` and therefore not in that
        set, so they are loaded here separately.
        """
        from vllm_neuron.utils.weight_loader import get_weight_loader

        for layer_id, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{layer_id}"
            self._load_scale_group(
                owner=layer.self_attn,
                sources=self._FP8_ATTN_SCALE_SOURCES,
                checkpoint=checkpoint,
                layer_prefix=prefix,
                tp_rank=tp_rank,
                device=device,
                get_loader=get_weight_loader,
            )
            self._load_scale_group(
                owner=layer.mlp,
                sources=self._FP8_MLP_SCALE_SOURCES,
                checkpoint=checkpoint,
                layer_prefix=prefix,
                tp_rank=tp_rank,
                device=device,
                get_loader=get_weight_loader,
            )

    @staticmethod
    def _load_scale_group(
        *,
        owner: nn.Module,
        sources: tuple[tuple[str, tuple[str, ...]], ...],
        checkpoint: "SafetensorsCheckpoint",
        layer_prefix: str,
        tp_rank: int,
        device: torch.device,
        get_loader,
    ) -> None:
        """Load one scale-buffer group (attention or MLP) on ``owner``."""
        for buffer_name, key_suffixes in sources:
            # Skip optional buffers this variant did not register.
            if not hasattr(owner, buffer_name):
                continue
            buf = getattr(owner, buffer_name)
            loader = get_loader(buf)
            slices = []
            for suffix in key_suffixes:
                full_key = f"{layer_prefix}.{suffix}"
                if full_key not in checkpoint._tensor_name_to_file:
                    raise KeyError(
                        f"Missing static-FP8 scale in checkpoint: {full_key!r}. "
                        "ModelOpt FP8 checkpoints must include per-projection "
                        "weight_scale and input_scale entries."
                    )
                slices.append(checkpoint._get_slice(full_key))
            tensor = loader.load(slices, tp_rank).to(device)
            setattr(owner, buffer_name, tensor)

    def _cast_to_model_dtype(self, rank_sharded: dict[str, torch.Tensor]) -> None:
        """Safety-net cast of tensors whose destination parameter is the
        model's compute dtype but whose loader returned a different dtype.

        fp8 weights and fp32 scale buffers keep their loader dtype. Gating
        on the destination parameter (rather than casting everything to
        ``config.torch_dtype``) is why this is safe for the quantized path.
        """
        target_dtype = self.config.torch_dtype
        for name, tensor in list(rank_sharded.items()):
            dest = self._lookup_parameter(name)
            if dest is None:
                # Legacy fallback (no destination param found).
                if tensor.dtype != target_dtype:
                    rank_sharded[name] = tensor.to(target_dtype)
                continue
            if dest.dtype == target_dtype and tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

    def _lookup_parameter(self, dotted_name: str) -> torch.Tensor | None:
        """Return the Parameter/Buffer at ``dotted_name`` on ``self``, or
        ``None`` if any component of the path is missing."""
        obj: object = self
        for part in dotted_name.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor):
            return obj
        return None

    # ------------------------------------------------------------------
    # Weight Loading
    # ------------------------------------------------------------------

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from a Qwen3 ModelOpt static-FP8 checkpoint.

        The pipelined loader loads parameters (fp8 weights and bf16
        norms); the static-FP8 scale buffers are non-persistent and are
        loaded by :meth:`_load_static_fp8_scales`.
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings: dict[str, str | list[str]] = {}

        # Embedding and LM head.
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["model.norm.weight"] = "model.norm.weight"

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"

            # Separate Q, K, V -> fused QKV.
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )

            # QK-norm weights (per-head, not TP-sharded).
            mappings[f"{prefix}.self_attn.q_norm.weight"] = (
                f"{prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{prefix}.self_attn.k_norm.weight"] = (
                f"{prefix}.self_attn.k_norm.weight"
            )

            # Layer norm names.
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            # Dense MLP weight names.
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        # Static-FP8 scales are non-persistent buffers: load them separately.
        self._load_static_fp8_scales(checkpoint, tp_rank, device)

        # Keep fp8 weights and fp32 scales; cast only ordinary bf16 tensors.
        self._cast_to_model_dtype(rank_sharded)

        self._load_kv_cache_scales(checkpoint, device)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
