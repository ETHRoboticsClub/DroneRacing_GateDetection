#!/usr/bin/env python3
"""Export the multi-task keypoint checkpoint to ONNX, build an FP16 TensorRT
engine, verify numerical correctness against torch, and report latency.

Sibling of ``scripts/bench_trt.py`` (which handles single-head seg-only
checkpoints — report 03's canonical numbers). Kept separate so the seg-only
path stays byte-identical to its report-03 form, not adapted around the
new dict-output shape.

What this script answers:
  1. Does the multi-task checkpoint export cleanly to ONNX (opset 18)?
  2. Does TRT 10.x build a FP16 engine from that ONNX?
  3. Do the TRT outputs match torch's outputs within FP16 tolerance?
  4. What is the per-frame latency for the multi-output model?

The numerical correctness check (#3) is the load-bearing one — a built
engine that emits wrong outputs is worse than no engine at all
(CLAUDE.md §0 / §6).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _flop_estimate(model: nn.Module, shape: tuple[int, ...]) -> str:
    """Return a string with FLOP estimate, or '(thop unavailable)'."""
    try:
        from thop import profile
        x = torch.zeros(*shape, device=next(model.parameters()).device)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return f"{flops / 1e9:.2f} GFLOPs ({params/1e6:.1f} M params)"
    except Exception as e:
        return f"(thop unavailable: {e})"


class KeypointExportWrapper(nn.Module):
    """Adapt the multi-task model to a fixed (seg, kp) tuple output.

    Inner model (``_DeepLabV3PlusWithKeypoint``) returns a Python ``dict``
    that the ONNX legacy tracer can emit but downstream TRT cannot. We
    re-route through the outer ``SegmentationModelWrapper.forward_multi``
    and unpack into a deterministic 2-tuple.
    """

    def __init__(self, outer_wrapper: nn.Module):
        super().__init__()
        self.outer = outer_wrapper

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_tuple, kp = self.outer.forward_multi(x)
        # Last seg head output is the full-resolution map per the
        # SegmentationModelWrapper contract; for DeepLabV3+ there is only one.
        return seg_tuple[-1], kp


def export_onnx(checkpoint: Path, onnx_path: Path, img_size: int = 384) -> tuple[int, str, KeypointExportWrapper]:
    """Load the multi-task checkpoint and export to ONNX with two outputs."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gate_detection.inference.predictor import load_checkpoint

    print(f"[export] loading {checkpoint}", flush=True)
    model, ckpt_meta = load_checkpoint(str(checkpoint), device="cuda")

    mc = ckpt_meta.get("model_config", {})
    if not mc.get("keypoint_head", False):
        raise RuntimeError(
            f"checkpoint {checkpoint} has model_config.keypoint_head=False — "
            "use scripts/bench_trt.py for the seg-only path."
        )
    if mc.get("name") != "deeplabv3plus":
        raise RuntimeError(
            f"checkpoint {checkpoint} is not a deeplabv3plus model "
            f"(model_config.name={mc.get('name')!r}); the keypoint head is "
            "only wired for deeplabv3plus today."
        )

    wrapped = KeypointExportWrapper(model).cuda().eval()
    flops_str = _flop_estimate(wrapped, (1, 3, img_size, img_size))
    print(f"[export] flops: {flops_str}", flush=True)

    dummy = torch.randn(1, 3, img_size, img_size, device="cuda")
    print(f"[export] writing {onnx_path}", flush=True)
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=["mask", "keypoint"],
        dynamic_axes={
            "input":    {0: "batch"},
            "mask":     {0: "batch"},
            "keypoint": {0: "batch"},
        },
        do_constant_folding=True,
    )
    return onnx_path.stat().st_size, flops_str, wrapped


def build_engine(onnx_path: Path, engine_path: Path, *, fp16: bool, max_workspace_gb: int = 4) -> tuple[float, int]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit-batch (default in TRT 10.x)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] parsing {onnx_path}", flush=True)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}", flush=True)
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled", flush=True)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 384, 384), (1, 3, 384, 384), (1, 3, 384, 384))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None")
    build_s = time.perf_counter() - t0
    engine_bytes = bytes(serialized)
    engine_path.write_bytes(engine_bytes)
    return build_s, len(engine_bytes)


def numerical_check(
    engine_path: Path,
    torch_model: nn.Module,
    *,
    img_size: int = 384,
    seed: int = 42,
) -> dict:
    """Compare TRT outputs vs torch outputs on a fixed input.

    Critical because "engine built" ≠ "engine correct". Reports per-output
    max-absolute-error and per-output mean-absolute-error against the torch
    baseline. FP16 should agree to within a few ×1e-3 on activations bounded
    in [0, 1] (sigmoid outputs); larger errors indicate an op-coverage gap
    or an incorrect dynamic-axis binding.
    """
    import tensorrt as trt

    # Uniform [0, 1] input mimics the deployment-time pixel distribution
    # (normalised uint8). Out-of-distribution Gaussian noise pushes both
    # sigmoid heads into saturation (output ≈ 0 everywhere) and the
    # absolute-error check then collapses to 0 - 0 = 0, hiding any real
    # FP16 op-coverage gap. CLAUDE.md §6 — no silent passes.
    torch.manual_seed(seed)
    inp_torch = torch.rand(1, 3, img_size, img_size, device="cuda")
    with torch.no_grad():
        seg_t, kp_t = torch_model(inp_torch)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    # IO bindings — discover by name rather than index for safety against
    # ONNX export reordering.
    name_to_mode = {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        name_to_mode[n] = engine.get_tensor_mode(n)
    print(f"[check]  TRT IO tensors: {list(name_to_mode.keys())}", flush=True)

    in_shape  = (1, 3, img_size, img_size)
    seg_shape = (1, 1, img_size, img_size)
    kp_shape  = (1, 4, img_size, img_size)
    context.set_input_shape("input", in_shape)

    inp_trt = inp_torch.contiguous()
    seg_trt = torch.empty(*seg_shape, dtype=torch.float32, device="cuda").contiguous()
    kp_trt  = torch.empty(*kp_shape,  dtype=torch.float32, device="cuda").contiguous()
    context.set_tensor_address("input",    int(inp_trt.data_ptr()))
    context.set_tensor_address("mask",     int(seg_trt.data_ptr()))
    context.set_tensor_address("keypoint", int(kp_trt.data_ptr()))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
    torch.cuda.synchronize()

    seg_diff = (seg_trt - seg_t).abs()
    kp_diff  = (kp_trt  - kp_t).abs()
    return {
        "seg_max_abs_err":  float(seg_diff.max()),
        "seg_mean_abs_err": float(seg_diff.mean()),
        "kp_max_abs_err":   float(kp_diff.max()),
        "kp_mean_abs_err":  float(kp_diff.mean()),
        "seg_torch_range":  (float(seg_t.min()), float(seg_t.max())),
        "kp_torch_range":   (float(kp_t.min()),  float(kp_t.max())),
    }


def benchmark(engine_path: Path, *, n_warmup: int = 20, n_iters: int = 200, img_size: int = 384) -> dict:
    """N-iter latency bench with both outputs bound."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    in_shape  = (1, 3, img_size, img_size)
    seg_shape = (1, 1, img_size, img_size)
    kp_shape  = (1, 4, img_size, img_size)
    context.set_input_shape("input", in_shape)

    inp = torch.randn(*in_shape,  dtype=torch.float32, device="cuda").contiguous()
    seg = torch.empty(*seg_shape, dtype=torch.float32, device="cuda").contiguous()
    kp  = torch.empty(*kp_shape,  dtype=torch.float32, device="cuda").contiguous()
    context.set_tensor_address("input",    int(inp.data_ptr()))
    context.set_tensor_address("mask",     int(seg.data_ptr()))
    context.set_tensor_address("keypoint", int(kp.data_ptr()))

    stream = torch.cuda.Stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        for i in range(n_iters):
            starts[i].record(stream)
            context.execute_async_v3(stream.cuda_stream)
            ends[i].record(stream)
    torch.cuda.synchronize()

    times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {
        "n":      int(n_iters),
        "mean":   float(times_ms.mean()),
        "median": float(np.median(times_ms)),
        "p99":    float(np.percentile(times_ms, 99)),
        "min":    float(times_ms.min()),
        "max":    float(times_ms.max()),
        "std":    float(times_ms.std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_deeplabv3plus_keypoint.pth")
    p.add_argument("--onnx",       default="checkpoints/gate_detection_keypoint.onnx")
    p.add_argument("--engine",     default="checkpoints/gate_detection_keypoint_fp16.engine")
    p.add_argument("--n-iters",    type=int, default=200)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)

    onnx_path   = Path(args.onnx)
    engine_path = Path(args.engine)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_bytes, flops_str, torch_model = export_onnx(Path(args.checkpoint), onnx_path)
    print(f"[export] OK — {onnx_bytes/1e6:.1f} MB ONNX", flush=True)

    build_s, eng_bytes = build_engine(onnx_path, engine_path, fp16=True)
    print(f"[build]  OK — {eng_bytes/1e6:.1f} MB engine, build {build_s:.1f} s", flush=True)

    diff = numerical_check(engine_path, torch_model)
    print(f"\n[check]  torch vs TRT-FP16 outputs", flush=True)
    print(f"  seg  range torch        : [{diff['seg_torch_range'][0]:+.4f}, {diff['seg_torch_range'][1]:+.4f}]", flush=True)
    print(f"  seg  max  |trt - torch| : {diff['seg_max_abs_err']:.5f}", flush=True)
    print(f"  seg  mean |trt - torch| : {diff['seg_mean_abs_err']:.5f}", flush=True)
    print(f"  kp   range torch        : [{diff['kp_torch_range'][0]:+.4f}, {diff['kp_torch_range'][1]:+.4f}]", flush=True)
    print(f"  kp   max  |trt - torch| : {diff['kp_max_abs_err']:.5f}", flush=True)
    print(f"  kp   mean |trt - torch| : {diff['kp_mean_abs_err']:.5f}", flush=True)
    seg_ok = diff["seg_max_abs_err"] < 5e-3
    kp_ok  = diff["kp_max_abs_err"]  < 5e-3
    if not (seg_ok and kp_ok):
        print(
            "\n[check]  WARN: max-abs-err exceeds 5e-3 on at least one output. "
            "FP16 sigmoid outputs in [0,1] should agree to ~1e-3; larger "
            "differences suggest an op-coverage gap (e.g. Resize alignment, "
            "BatchNorm folding) — investigate before deploying.",
            flush=True,
        )
    else:
        print("\n[check]  OK — both outputs within FP16 tolerance.", flush=True)

    bench = benchmark(engine_path, n_iters=args.n_iters)
    fps = 1000.0 / bench["mean"]
    print(f"\n[bench]  N={bench['n']} iters, FP16, {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  latency_mean   : {bench['mean']:6.2f} ms   ({fps:6.1f} fps)", flush=True)
    print(f"  latency_median : {bench['median']:6.2f} ms", flush=True)
    print(f"  latency_p99    : {bench['p99']:6.2f} ms", flush=True)
    print(f"  latency_min    : {bench['min']:6.2f} ms", flush=True)
    print(f"  latency_max    : {bench['max']:6.2f} ms", flush=True)
    print(f"  latency_std    : {bench['std']:6.2f} ms", flush=True)
    print(f"  flops          : {flops_str}", flush=True)


if __name__ == "__main__":
    main()
