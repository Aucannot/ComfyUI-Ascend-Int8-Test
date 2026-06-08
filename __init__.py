import importlib
import logging
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import comfy.ops
import comfy.sd
from comfy.comfy_types.node_typing import IO
import folder_paths


BACKENDS = [
    "auto_torch_npu_with_fallback",
    "torch_npu_strict",
    "fallback_dequant_only",
]

SCALE_MODES = [
    "per_channel",
    "per_tensor",
]

_RUNTIME_STATS = {}


def _reset_runtime_stats():
    _RUNTIME_STATS.clear()
    _RUNTIME_STATS.update(
        {
            "linear_forward_calls": 0,
            "torch_npu_int8_calls": 0,
            "auto_fallback_dequant_calls": 0,
            "forced_dequant_calls": 0,
            "weight_patch_dequant_calls": 0,
            "nan_input_calls": 0,
            "inf_input_calls": 0,
            "nan_output_calls": 0,
            "inf_output_calls": 0,
            "last_error": "",
            "layers": {},
        }
    )


def _bump_runtime(key: str, layer_name: str = ""):
    _RUNTIME_STATS[key] = _RUNTIME_STATS.get(key, 0) + 1
    if layer_name:
        layer_stats = _RUNTIME_STATS.setdefault("layers", {}).setdefault(
            layer_name,
            {
                "calls": 0,
                "torch_npu_int8": 0,
                "auto_fallback_dequant": 0,
                "forced_dequant": 0,
                "weight_patch_dequant": 0,
            },
        )
        layer_stats["calls"] += 1
        if key == "torch_npu_int8_calls":
            layer_stats["torch_npu_int8"] += 1
        elif key == "auto_fallback_dequant_calls":
            layer_stats["auto_fallback_dequant"] += 1
        elif key == "forced_dequant_calls":
            layer_stats["forced_dequant"] += 1
        elif key == "weight_patch_dequant_calls":
            layer_stats["weight_patch_dequant"] += 1


def _record_tensor_health(prefix: str, tensor: torch.Tensor):
    if tensor is None or not tensor.dtype.is_floating_point:
        return
    try:
        if torch.isnan(tensor).any().item():
            _RUNTIME_STATS[f"nan_{prefix}_calls"] = _RUNTIME_STATS.get(f"nan_{prefix}_calls", 0) + 1
        if torch.isinf(tensor).any().item():
            _RUNTIME_STATS[f"inf_{prefix}_calls"] = _RUNTIME_STATS.get(f"inf_{prefix}_calls", 0) + 1
    except Exception as e:
        _RUNTIME_STATS["last_error"] = f"tensor health check failed: {e.__class__.__name__}: {e}"


def _format_runtime_stats() -> str:
    layer_lines = []
    layers = _RUNTIME_STATS.get("layers", {})
    for name, stats in sorted(layers.items(), key=lambda item: item[1]["calls"], reverse=True)[:40]:
        layer_lines.append(
            f"  {name}: calls={stats['calls']}, "
            f"torch_npu_int8={stats['torch_npu_int8']}, "
            f"auto_fallback_dequant={stats['auto_fallback_dequant']}, "
            f"forced_dequant={stats['forced_dequant']}, "
            f"weight_patch_dequant={stats['weight_patch_dequant']}"
        )
    if len(layers) > 40:
        layer_lines.append(f"  ... {len(layers) - 40} more")
    if not layer_lines:
        layer_lines.append("  -")

    return "\n".join(
        [
            "Ascend INT8 runtime report",
            f"linear_forward_calls: {_RUNTIME_STATS.get('linear_forward_calls', 0)}",
            f"torch_npu_int8_calls: {_RUNTIME_STATS.get('torch_npu_int8_calls', 0)}",
            f"auto_fallback_dequant_calls: {_RUNTIME_STATS.get('auto_fallback_dequant_calls', 0)}",
            f"forced_dequant_calls: {_RUNTIME_STATS.get('forced_dequant_calls', 0)}",
            f"weight_patch_dequant_calls: {_RUNTIME_STATS.get('weight_patch_dequant_calls', 0)}",
            f"nan_input_calls: {_RUNTIME_STATS.get('nan_input_calls', 0)}",
            f"inf_input_calls: {_RUNTIME_STATS.get('inf_input_calls', 0)}",
            f"nan_output_calls: {_RUNTIME_STATS.get('nan_output_calls', 0)}",
            f"inf_output_calls: {_RUNTIME_STATS.get('inf_output_calls', 0)}",
            f"last_error: {_RUNTIME_STATS.get('last_error', '') or '-'}",
            "top_layers:",
            *layer_lines,
        ]
    )


_reset_runtime_stats()


@dataclass
class Int8LoadStats:
    seen_linear: int = 0
    quantized_linear: int = 0
    skipped_linear: int = 0
    original_weight_bytes: int = 0
    quantized_weight_bytes: int = 0
    quantized_layers: list[str] = field(default_factory=list)
    skipped_layers: list[str] = field(default_factory=list)


@dataclass
class Int8Config:
    backend: str
    scale_mode: str
    include_regex: str
    exclude_regex: str
    min_in_features: int
    min_out_features: int
    stats: Int8LoadStats = field(default_factory=Int8LoadStats)

    def __post_init__(self):
        self.include_re = re.compile(self.include_regex or ".*")
        self.exclude_re = re.compile(self.exclude_regex) if self.exclude_regex else None

    def should_quantize(self, layer_name: str, in_features: int, out_features: int) -> tuple[bool, str]:
        self.stats.seen_linear += 1
        if in_features < self.min_in_features:
            return False, f"in_features<{self.min_in_features}"
        if out_features < self.min_out_features:
            return False, f"out_features<{self.min_out_features}"
        if not self.include_re.search(layer_name):
            return False, "include_regex"
        if self.exclude_re is not None and self.exclude_re.search(layer_name):
            return False, "exclude_regex"
        return True, "quantized"


def _dtype_nbytes(dtype: torch.dtype) -> int:
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        return 4


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * _dtype_nbytes(tensor.dtype)


def _short_list(items: list[str], limit: int = 40) -> str:
    if not items:
        return "-"
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f"\n  ... {len(items) - limit} more"
    return "\n  " + "\n  ".join(shown) + suffix


def _safe_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as e:
        return f"not importable: {e.__class__.__name__}: {e}"


def _torch_npu_module():
    try:
        return importlib.import_module("torch_npu")
    except Exception:
        return None


def _linear_weight_quant_op():
    torch_npu = _torch_npu_module()
    if torch_npu is None:
        return None
    return getattr(torch_npu, "npu_weight_quant_batchmatmul", None)


def _linear_weight_quant_module_available() -> bool:
    try:
        module = importlib.import_module("torch_npu.contrib.module")
        return hasattr(module, "LinearWeightQuant")
    except Exception:
        return False


def _quantize_weight_to_int8(weight: torch.Tensor, scale_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    # torch_npu.npu_weight_quant_batchmatmul computes x @ antiquant(weight).
    # F.linear computes x @ original_weight.T, so store original_weight.T as K,N.
    weight_t = torch.nan_to_num(weight.detach().to(torch.float32).t().contiguous())
    eps = torch.tensor(1.0e-8, dtype=torch.float32, device=weight_t.device)

    if scale_mode == "per_tensor":
        scale = torch.clamp(weight_t.abs().amax() / 127.0, min=eps)
        qweight = torch.clamp(torch.round(weight_t / scale), -127, 127).to(torch.int8)
        return qweight.contiguous(), scale.reshape(1).contiguous()

    if scale_mode != "per_channel":
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")

    scale = torch.clamp(weight_t.abs().amax(dim=0) / 127.0, min=eps)
    qweight = torch.clamp(torch.round(weight_t / scale), -127, 127).to(torch.int8)
    return qweight.contiguous(), scale.contiguous()


def _dequantized_linear(
    x_2d: torch.Tensor,
    qweight: torch.Tensor,
    antiquant_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    scale = antiquant_scale.to(device=x_2d.device, dtype=torch.float32)
    q = qweight.to(device=x_2d.device)
    weight_t = q.to(torch.float32) * scale
    weight = weight_t.t().to(dtype=x_2d.dtype)
    b = bias.to(device=x_2d.device, dtype=x_2d.dtype) if bias is not None else None
    return torch.nn.functional.linear(x_2d, weight, b)


def _torch_npu_weight_quant_linear(
    x_2d: torch.Tensor,
    qweight: torch.Tensor,
    antiquant_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    op = _linear_weight_quant_op()
    if op is None:
        raise RuntimeError("torch_npu.npu_weight_quant_batchmatmul is not available")
    if x_2d.device.type != "npu":
        raise RuntimeError(f"input device is {x_2d.device.type}, expected npu")

    qw = qweight.to(device=x_2d.device)
    scale = antiquant_scale.to(device=x_2d.device, dtype=torch.float32)
    b = bias.to(device=x_2d.device, dtype=x_2d.dtype) if bias is not None else None

    try:
        return op(x_2d, qw, scale, None, None, None, b, 0, 0)
    except TypeError:
        return op(
            x_2d,
            qw,
            scale,
            antiquant_offset=None,
            quant_scale=None,
            quant_offset=None,
            bias=b,
            antiquant_group_size=0,
            inner_precise=0,
        )


def _format_stats(config: Int8Config, title: str) -> str:
    stats = config.stats
    saved = stats.original_weight_bytes - stats.quantized_weight_bytes
    ratio = 0.0
    if stats.original_weight_bytes:
        ratio = stats.quantized_weight_bytes / stats.original_weight_bytes

    return "\n".join(
        [
            title,
            f"backend: {config.backend}",
            f"scale_mode: {config.scale_mode}",
            f"include_regex: {config.include_regex or '.*'}",
            f"exclude_regex: {config.exclude_regex or '-'}",
            f"min_in_features: {config.min_in_features}",
            f"min_out_features: {config.min_out_features}",
            f"linear_seen: {stats.seen_linear}",
            f"linear_quantized: {stats.quantized_linear}",
            f"linear_skipped: {stats.skipped_linear}",
            f"original_linear_weight_bytes: {stats.original_weight_bytes}",
            f"int8_weight_plus_scale_bytes: {stats.quantized_weight_bytes}",
            f"rough_weight_ratio: {ratio:.4f}",
            f"rough_bytes_saved: {saved}",
            "quantized_layers:" + _short_list(stats.quantized_layers),
            "skipped_layers:" + _short_list(stats.skipped_layers),
        ]
    )


def make_ascend_int8_ops(config: Int8Config):
    class AscendInt8Ops(comfy.ops.manual_cast):
        class Linear(comfy.ops.manual_cast.Linear):
            comfy_cast_weights = True

            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
                self._ascend_int8_enabled = False
                self._ascend_int8_config = config
                self._ascend_int8_layer_name = ""

            def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                layer_name = prefix.rstrip(".")
                should_quantize, reason = config.should_quantize(layer_name, self.in_features, self.out_features)
                if not should_quantize:
                    config.stats.skipped_linear += 1
                    config.stats.skipped_layers.append(f"{layer_name}: {reason}")
                    return super()._load_from_state_dict(
                        state_dict,
                        prefix,
                        local_metadata,
                        strict,
                        missing_keys,
                        unexpected_keys,
                        error_msgs,
                    )

                weight_key = f"{prefix}weight"
                bias_key = f"{prefix}bias"
                weight = state_dict.pop(weight_key, None)
                bias = state_dict.pop(bias_key, None)

                if weight is None:
                    config.stats.skipped_linear += 1
                    config.stats.skipped_layers.append(f"{layer_name}: missing weight")
                    missing_keys.append(weight_key)
                    return

                try:
                    qweight, antiquant_scale = _quantize_weight_to_int8(weight, config.scale_mode)
                except Exception as e:
                    config.stats.skipped_linear += 1
                    config.stats.skipped_layers.append(f"{layer_name}: quantize failed: {e}")
                    error_msgs.append(f"{layer_name}: quantize failed: {traceback.format_exc()}")
                    return

                config.stats.quantized_linear += 1
                config.stats.original_weight_bytes += _tensor_nbytes(weight)
                config.stats.quantized_weight_bytes += _tensor_nbytes(qweight) + _tensor_nbytes(antiquant_scale)
                config.stats.quantized_layers.append(layer_name)

                self.weight = None
                self.register_buffer("qweight", qweight, persistent=True)
                self.register_buffer("antiquant_scale", antiquant_scale, persistent=True)

                if bias is not None:
                    self.bias = torch.nn.Parameter(bias.detach().clone(), requires_grad=False)
                else:
                    self.bias = None

                self._ascend_int8_enabled = True
                self._ascend_int8_layer_name = layer_name

            def _bias_for_input(self, input: torch.Tensor) -> Optional[torch.Tensor]:
                bias = self.bias
                if bias is not None:
                    for f in self.bias_function:
                        bias = f(bias)
                    bias = bias.to(device=input.device, dtype=input.dtype)
                return bias

            def _dequant_weight_for_input(self, input: torch.Tensor) -> torch.Tensor:
                scale = self.antiquant_scale.to(device=input.device, dtype=torch.float32)
                q = self.qweight.to(device=input.device)
                weight_t = q.to(torch.float32) * scale
                return weight_t.t().to(dtype=input.dtype)

            def forward(self, input, *args, **kwargs):
                if not self._ascend_int8_enabled:
                    return super().forward(input, *args, **kwargs)

                comfy.ops.run_every_op()
                _RUNTIME_STATS["linear_forward_calls"] = _RUNTIME_STATS.get("linear_forward_calls", 0) + 1
                _record_tensor_health("input", input)

                input_shape = tuple(input.shape)
                if input.ndim < 2:
                    raise RuntimeError(
                        f"Ascend INT8 Linear expects input rank >= 2, got {input.ndim} "
                        f"at {self._ascend_int8_layer_name}"
                    )
                x_2d = input.reshape(-1, input_shape[-1])
                bias = self._bias_for_input(input)

                # Weight patch functions, such as LoRA, need a real dequantized weight.
                if len(self.weight_function) > 0 or len(self.bias_function) > 0:
                    _bump_runtime("weight_patch_dequant_calls", self._ascend_int8_layer_name)
                    weight = self._dequant_weight_for_input(input)
                    for f in self.weight_function:
                        weight = f(weight)
                    out = torch.nn.functional.linear(x_2d, weight, bias)
                elif config.backend == "fallback_dequant_only":
                    _bump_runtime("forced_dequant_calls", self._ascend_int8_layer_name)
                    out = _dequantized_linear(x_2d, self.qweight, self.antiquant_scale, bias)
                elif config.backend == "torch_npu_strict":
                    out = _torch_npu_weight_quant_linear(x_2d, self.qweight, self.antiquant_scale, bias)
                    _bump_runtime("torch_npu_int8_calls", self._ascend_int8_layer_name)
                else:
                    try:
                        out = _torch_npu_weight_quant_linear(x_2d, self.qweight, self.antiquant_scale, bias)
                        _bump_runtime("torch_npu_int8_calls", self._ascend_int8_layer_name)
                    except Exception as e:
                        _RUNTIME_STATS["last_error"] = f"{self._ascend_int8_layer_name}: {e.__class__.__name__}: {e}"
                        _bump_runtime("auto_fallback_dequant_calls", self._ascend_int8_layer_name)
                        logging.debug(
                            "Ascend INT8 fallback for %s: %s",
                            self._ascend_int8_layer_name,
                            e,
                        )
                        out = _dequantized_linear(x_2d, self.qweight, self.antiquant_scale, bias)

                _record_tensor_health("output", out)
                return out.reshape(input_shape[:-1] + (self.out_features,))

            def extra_repr(self):
                base = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
                if self._ascend_int8_enabled:
                    return base + f", ascend_int8=True, backend={config.backend}, scale_mode={config.scale_mode}"
                return base

    return AscendInt8Ops


def _make_config(backend, scale_mode, include_regex, exclude_regex, min_in_features, min_out_features) -> Int8Config:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")
    return Int8Config(
        backend=backend,
        scale_mode=scale_mode,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        min_in_features=int(min_in_features),
        min_out_features=int(min_out_features),
    )


def _load_model_options(config: Int8Config) -> dict[str, Any]:
    return {"custom_operations": make_ascend_int8_ops(config)}


class AscendInt8EnvironmentReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "probe_torch_npu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "ascend/int8"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, probe_torch_npu=True):
        lines = [
            "Ascend INT8 environment report",
            f"python: {sys.executable}",
            f"torch: {torch.__version__}",
            f"torch_cuda: {torch.version.cuda}",
            f"torch_npu: {_safe_version('torch_npu') if probe_torch_npu else 'not probed'}",
            f"has torch.npu: {hasattr(torch, 'npu')}",
        ]

        if hasattr(torch, "npu"):
            try:
                lines.append(f"torch.npu.is_available: {torch.npu.is_available()}")
                lines.append(f"torch.npu.device_count: {torch.npu.device_count()}")
                if torch.npu.is_available():
                    lines.append(f"current_npu: {torch.npu.current_device()}")
                    lines.append(f"npu_name: {torch.npu.get_device_name(torch.npu.current_device())}")
            except Exception as e:
                lines.append(f"torch.npu probe_error: {e.__class__.__name__}: {e}")

        if probe_torch_npu:
            lines.extend(
                [
                    f"npu_weight_quant_batchmatmul: {_linear_weight_quant_op() is not None}",
                    f"npu_anti_quant: {hasattr(_torch_npu_module(), 'npu_anti_quant') if _torch_npu_module() else False}",
                    f"LinearWeightQuant: {_linear_weight_quant_module_available()}",
                ]
            )

        report = "\n".join(lines)
        return {"ui": {"text": (report,)}, "result": (report,)}


class AscendInt8RuntimeReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (IO.ANY, {}),
                "reset_after_report": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "ascend/int8"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, source=None, reset_after_report=False):
        report = _format_runtime_stats()
        if reset_after_report:
            _reset_runtime_stats()
        return {"ui": {"text": (report,)}, "result": (report,)}


class AscendInt8UNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "backend": (BACKENDS, {"default": "auto_torch_npu_with_fallback"}),
                "scale_mode": (SCALE_MODES, {"default": "per_channel"}),
                "include_regex": ("STRING", {"default": ".*", "multiline": False}),
                "exclude_regex": ("STRING", {"default": "", "multiline": False}),
                "min_in_features": ("INT", {"default": 16, "min": 1, "max": 65536}),
                "min_out_features": ("INT", {"default": 16, "min": 1, "max": 65536}),
                "reset_runtime_stats": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_unet"
    CATEGORY = "ascend/int8"

    def load_unet(
        self,
        unet_name,
        backend,
        scale_mode,
        include_regex,
        exclude_regex,
        min_in_features,
        min_out_features,
        reset_runtime_stats=True,
    ):
        if reset_runtime_stats:
            _reset_runtime_stats()
        config = _make_config(backend, scale_mode, include_regex, exclude_regex, min_in_features, min_out_features)
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=_load_model_options(config))
        report = _format_stats(config, f"Ascend INT8 UNET load report: {unet_name}")
        return (model, report)


class AscendInt8CheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "backend": (BACKENDS, {"default": "auto_torch_npu_with_fallback"}),
                "scale_mode": (SCALE_MODES, {"default": "per_channel"}),
                "include_regex": ("STRING", {"default": ".*", "multiline": False}),
                "exclude_regex": ("STRING", {"default": "", "multiline": False}),
                "min_in_features": ("INT", {"default": 16, "min": 1, "max": 65536}),
                "min_out_features": ("INT", {"default": 16, "min": 1, "max": 65536}),
                "reset_runtime_stats": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "report")
    FUNCTION = "load_checkpoint"
    CATEGORY = "ascend/int8"

    def load_checkpoint(
        self,
        ckpt_name,
        backend,
        scale_mode,
        include_regex,
        exclude_regex,
        min_in_features,
        min_out_features,
        reset_runtime_stats=True,
    ):
        if reset_runtime_stats:
            _reset_runtime_stats()
        config = _make_config(backend, scale_mode, include_regex, exclude_regex, min_in_features, min_out_features)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=_load_model_options(config),
        )
        report = _format_stats(config, f"Ascend INT8 checkpoint load report: {ckpt_name}")
        return (model, clip, vae, report)


NODE_CLASS_MAPPINGS = {
    "AscendInt8EnvironmentReport": AscendInt8EnvironmentReport,
    "AscendInt8RuntimeReport": AscendInt8RuntimeReport,
    "AscendInt8UNETLoader": AscendInt8UNETLoader,
    "AscendInt8CheckpointLoader": AscendInt8CheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AscendInt8EnvironmentReport": "Ascend INT8 Environment Report",
    "AscendInt8RuntimeReport": "Ascend INT8 Runtime Report",
    "AscendInt8UNETLoader": "Load Diffusion Model (Ascend INT8)",
    "AscendInt8CheckpointLoader": "Load Checkpoint (Ascend INT8)",
}
