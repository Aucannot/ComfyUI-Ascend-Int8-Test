# ComfyUI Ascend INT8 Validation Nodes

This custom node pack is for deployment validation of ComfyUI + Ascend NPU + INT8
linear weight quantization.

It is intentionally safe on non-Ascend machines:

- NVIDIA/local development can import the nodes and use `fallback_dequant_only`.
- Ascend/FaaS deployment can use `auto_torch_npu_with_fallback` first.
- `torch_npu_strict` raises at runtime if `torch_npu.npu_weight_quant_batchmatmul`
  is missing, the input is not on `npu`, or the Ascend op fails.

## Nodes

- `Ascend INT8 Environment Report`
  - Reports Python, torch, torch_npu, `torch.npu`, and required INT8 APIs.

- `Load Diffusion Model (Ascend INT8)`
  - Loads a file from `models/diffusion_models`.
  - Quantizes eligible `Linear` weights to int8 while the model is loaded.

- `Load Checkpoint (Ascend INT8)`
  - Loads a normal checkpoint from `models/checkpoints`.
  - Quantizes the diffusion model Linear weights only. CLIP and VAE are left on
    normal ComfyUI paths.

## Backend Modes

- `auto_torch_npu_with_fallback`
  - Tries `torch_npu.npu_weight_quant_batchmatmul` on NPU inputs.
  - Falls back to dequantized `torch.nn.functional.linear` otherwise.

- `torch_npu_strict`
  - Requires the Ascend INT8 op. Use this for FaaS validation after basic import
    succeeds.

- `fallback_dequant_only`
  - Never calls `torch_npu`; useful for local NVIDIA smoke tests.

## Current Scope

This is a validation plugin, not a production quantization implementation.

Known limitations:

- Only `Linear` layers are quantized.
- Weight quantization is online at model load time, so peak memory is not optimal.
- LoRA or other weight patches fall back to dequantized linear for correctness.
- Text encoders and VAE are not quantized by these loader nodes.
- Per-channel scale means one scale per output channel.

The main integration point for a custom quantization library is the INT8 matmul
path in `__init__.py`.
