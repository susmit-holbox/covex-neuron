# SPDX-License-Identifier: Apache-2.0
"""Qwen3 checkpoint quantization parsing (MXFP8 and static FP8)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Any

# Canonical MX group size (E8M0 shared exponent per 32 elements). Imported by
# weight_loaders_mxfp8 as MX_GROUP_SIZE so the contract is single-sourced.
MXFP8_GROUP_SIZE = 32


class QuantScheme(str, Enum):
    NONE = "none"
    MXFP8_DYNAMIC_PER_TOKEN_CHANNELWISE = "mxfp8_dynamic_per_token_channelwise"
    # NVIDIA ModelOpt static FP8 (float8_e4m3fn) weight + activation, per-tensor
    # scales. Trn2-compatible (unlike MXFP8, which is Trn3-only).
    FP8_STATIC_PER_TENSOR = "fp8_static_per_tensor"


@dataclass(frozen=True)
class QuantizationSpec:
    """Producer-agnostic view of the checkpoint quantization contract."""

    linear_scheme: QuantScheme

    def get_scheme(self, layer_index: int | None, prefix: str) -> QuantScheme:
        del layer_index, prefix
        return self.linear_scheme

    @classmethod
    def from_hf_quantization_config(
        cls, quantization_config: dict[str, Any] | None
    ) -> "QuantizationSpec | None":
        if not quantization_config:
            return None
        if not isinstance(quantization_config, dict):
            raise ValueError(
                "Expected quantization_config to be a dict, got "
                f"{type(quantization_config).__name__}."
            )

        quant_method = str(quantization_config.get("quant_method", "")).lower()
        if not quant_method:
            return None
        if quant_method == "compressed-tensors":
            return _parse_compressed_tensors_mxfp8(quantization_config)
        if quant_method == "modelopt":
            return _parse_modelopt(quantization_config)

        raise ValueError(
            f"Unsupported quantization_config.quant_method={quant_method!r}. "
            "Qwen3 dense currently supports compressed-tensors "
            "static-weight / dynamic-activation MXFP8 checkpoints and "
            "ModelOpt static FP8 checkpoints."
        )


def _parse_modelopt(quantization_config: dict[str, Any]) -> QuantizationSpec:
    """Parse a ModelOpt-format static-FP8 ``quantization_config``.

    Mirrors ``vllm_neuron/model/llama3/quantization.py:_parse_modelopt``.
    Accepts the nested shape (and the legacy flat shape, like vLLM)::

        {
          "quant_method": "modelopt",
          "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["lm_head", ...]
          }
        }

    ``exclude_modules`` is validated (it must list ``"lm_head"`` literally or
    via a wildcard) to catch unsupported checkpoints. Per-module dispatch is
    deferred — the spec stores only the scheme.

    Raises:
        ValueError: when ``quant_algo`` is not FP8, or ``exclude_modules`` is
            malformed / does not exclude ``lm_head``.
    """
    inner = quantization_config.get("quantization")
    algo_src: dict[str, Any] = inner if isinstance(inner, dict) else quantization_config

    quant_algo = str(algo_src.get("quant_algo", "")).upper()
    exclude_modules_raw = algo_src.get("exclude_modules") or []

    if quant_algo != "FP8":
        raise ValueError(
            "Qwen3 currently only supports ModelOpt quant_algo='FP8', "
            f"got quant_algo={quant_algo!r}."
        )
    if not isinstance(exclude_modules_raw, (list, tuple)):
        raise ValueError(
            "ModelOpt quantization_config.exclude_modules must be a list, "
            f"got {type(exclude_modules_raw).__name__}."
        )

    patterns = tuple(str(m) for m in exclude_modules_raw)
    if not any(p == "lm_head" or fnmatch("lm_head", p) for p in patterns):
        raise ValueError(
            "ModelOpt FP8 checkpoints for Qwen3 must list 'lm_head' in "
            f"exclude_modules; got exclude_modules={list(patterns)!r}."
        )

    return QuantizationSpec(linear_scheme=QuantScheme.FP8_STATIC_PER_TENSOR)


def _parse_compressed_tensors_mxfp8(
    quantization_config: dict[str, Any],
) -> QuantizationSpec:
    fmt = str(quantization_config.get("format", "")).lower()
    # "mxfp8-quantized" is the canonical MX label; "float-quantized" is also
    # accepted because some llm-compressor MX exporters stamp the generic FP8
    # label even for group-32 MX weights. The group_size==32 + strategy==group
    # checks below still reject any genuinely non-MX float-quantized checkpoint.
    supported_formats = {"float-quantized", "mxfp8-quantized"}
    if fmt not in supported_formats:
        raise ValueError(
            "Qwen3-32B dense MXFP8 support requires "
            "quantization_config.format in "
            f"{sorted(supported_formats)!r}, got "
            f"{fmt!r}."
        )

    config_groups = quantization_config.get("config_groups")
    if not isinstance(config_groups, dict) or not config_groups:
        raise ValueError(
            "Qwen3-32B dense MXFP8 support requires a non-empty config_groups mapping."
        )

    for group_name, group in config_groups.items():
        if not isinstance(group, dict):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires each config_groups "
                f"entry to be a dict, got {type(group).__name__} for "
                f"{group_name!r}."
            )

        weights = group.get("weights")
        if not isinstance(weights, dict):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires weight quantization "
                f"config in {group_name!r}."
            )

        input_activations = group.get("input_activations")
        if not isinstance(input_activations, dict):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires input activation "
                f"quantization config in {group_name!r}."
            )

        if group.get("output_activations") is not None:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support does not accept output_activations "
                f"in {group_name!r}."
            )

        if (
            str(weights.get("type", "")).lower() != "float"
            or int(weights.get("num_bits", 0)) != 8
        ):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires float8 weights in "
                f"{group_name!r}."
            )
        if bool(weights.get("dynamic")):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires static weight "
                f"quantization in {group_name!r}."
            )
        if str(weights.get("strategy", "")).lower() != "group":
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires groupwise weights in "
                f"{group_name!r}."
            )
        if int(weights.get("group_size", 0)) != MXFP8_GROUP_SIZE:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires weight group_size=32 in "
                f"{group_name!r}."
            )
        scale_dtype = weights.get("scale_dtype")
        if scale_dtype is not None and str(scale_dtype).lower() not in {
            "uint8",
            "u8",
            "torch.uint8",
        }:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires weight scale_dtype=torch.uint8 "
                f"(E8M0 storage) when declared, got {scale_dtype!r} in {group_name!r}."
            )

        if (
            str(input_activations.get("type", "")).lower() != "float"
            or int(input_activations.get("num_bits", 0)) != 8
        ):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires float8 input "
                f"activations in {group_name!r}."
            )
        if not bool(input_activations.get("dynamic")):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires dynamic input "
                f"activations in {group_name!r}."
            )
        if str(input_activations.get("strategy", "")).lower() != "group":
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires groupwise input "
                f"activations in {group_name!r}."
            )
        if int(input_activations.get("group_size", 0)) != MXFP8_GROUP_SIZE:
            raise ValueError(
                "Qwen3-32B dense MXFP8 support requires input group_size=32 in "
                f"{group_name!r}."
            )

        targets = group.get("targets") or []
        if not any(str(target) == "Linear" for target in targets):
            raise ValueError(
                "Qwen3-32B dense MXFP8 support currently requires Linear "
                f"targets in {group_name!r}."
            )

    return QuantizationSpec(
        linear_scheme=QuantScheme.MXFP8_DYNAMIC_PER_TOKEN_CHANNELWISE
    )
