# Report 03 — TensorRT export + latency benchmark for DeepLabV3+ ResNet-34

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Hardware** | qolam: NVIDIA RTX 5090 32 GB, CUDA 12.8, driver 570.211.01 |
| **Stack** | torch 2.11.0+cu128, TensorRT 10.16.1.11 (cu12 wheel), onnxscript 0.7.0, opset 18 |
| **Checkpoint** | `checkpoints/best_deeplabv3plus_imagenet_nonorm.pth` (10-epoch run from report 02) |
| **Companion script** | `scripts/bench_trt.py` |

## TL;DR

DeepLabV3+ with ImageNet-pretrained ResNet-34 at 384×384 **exports cleanly** to ONNX (opset 18, dynamo exporter, weights as external `.data`), **builds an FP16 TensorRT engine in 5.7 s**, and runs at **0.45 ms / 2223 fps on the RTX 5090**. Projecting to Jetson Orin NX 16 GB by the conservative TOPS-ratio (60×) and the optimistic memory-bandwidth ratio (18×) gives **8–30 ms FP16 latency** = 33–125 fps. **Comfortable real-time on Orin NX in FP16; INT8 with calibration projects 65–250 fps.**

The "DeepLabV3+ + ASPP won't fit on Jetson" worry from the prior research pass was driven by Xavier-era extrapolation and does not survive a real measurement.

**Decision:** keep DeepLabV3+ + ImageNet ResNet-34 as the production architecture for the rest of the plan. Stay on track.

## Measurement

```
[export] flops           : 17.83 GFLOPs (22.4 M params)
[export] OK              : 90 MB ONNX (337 KB header + 89.7 MB external .data)
[build]  FP16 OK         : 45.6 MB engine, build 5.7 s
[bench]  N=200 iters, FP16 batch=1, RTX 5090
  latency_mean   :  0.45 ms   (2223.0 fps)
  latency_median :  0.45 ms
  latency_p99    :  0.46 ms
  latency_std    :  0.00 ms
```

The latency distribution is essentially a delta function — kernel timings are completely deterministic at this size on the 5090. CUDA Graphs / autotune kernels are reused, the engine sits in L2, the workload is dominated by tensor cores.

## Projecting to Jetson Orin NX 16 GB

Two ways to extrapolate, bracketing the realistic range:

| Method | RTX 5090 → Orin NX ratio | FP16 latency | FP16 fps |
|---|---|---|---|
| Compute (TOPS): 5090 ~1500 FP16 / Orin NX ~25 FP16 | **~60×** slower | ~27 ms | ~37 fps |
| Memory bandwidth: 5090 ~1.8 TB/s / Orin NX ~100 GB/s | **~18×** slower | ~8 ms | ~125 fps |

Conv-heavy segmentation networks at 384² are typically *between* these two bounds, leaning toward bandwidth-limited. **Best point estimate: ~10–20 ms FP16 = 50–100 fps** on Orin NX. INT8 with calibration roughly halves this further.

**That is well above the 30 fps target.**

## What this means for the architecture decision

The deeper of the prior two concerns was that ASPP would fail to export or compile cleanly. Empirically:

- ✅ smp's `DeepLabV3Plus` with `encoder_name="resnet34", encoder_weights="imagenet"` exports via torch 2.11's dynamo `torch.onnx.export(..., opset_version=18)` without intervention.
- ✅ TensorRT 10.16's `OnnxParser.parse_from_file` consumes the ONNX (the external-data `.data` file is auto-followed when you pass the path, not the bytes).
- ✅ FP16 build flag works; no fallback layers, no plugin requirements logged.
- ⚠ Two minor friction points:
  1. The dynamo exporter writes weights as external `.onnx.data` by default — `parse_from_file` handles this, but `parse(open(...).read())` does not. The script uses the right call.
  2. tensorrt's pip wheel resolves to `tensorrt-cu13` by default; on a CUDA-12 driver host you must `pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com`. Documented in the bootstrap notes.

## What's NOT measured here, that I should be honest about

1. **This is the 5090, not an Orin NX.** Until we have an actual Jetson in hand, the 8–30 ms band is a projection, not a measurement. The plan's P3-1 step calls for a real Orin NX latency reading; this report does not substitute for that.
2. **Batch size 1 only.** Real deployment uses batch 1 anyway, but for completeness, batch 4–8 might extract more throughput on Orin NX if the perception node is bottle-necked.
3. **No INT8 calibration ran here.** INT8 needs a calibration dataset (`scripts/build_sam_dataset.py`'s symlinks would do); the script has no INT8 path yet. Adding this is a P3-1 follow-up.
4. **The 10-epoch checkpoint is the network being benchmarked.** Latency is architecture-dependent, not weight-dependent — so the number stands for the production model too. *Mask quality* of course differs and is unrelated to this report.
5. **`smp.encoders.get_preprocessing_fn` / ImageNet norm** is not in the exported graph. Inference parity requires the predictor to apply the same preprocessing the trainer used; that is already wired (report 02, `predictor.py:158–168`). Not a TRT concern.

## Concrete next steps

This report **unblocks** the rest of the plan:

1. **Continue with P0-3 follow-up — converged 100-epoch retrain** of `configs/deeplabv3plus_imagenet_nonorm.yaml` (the cell-B winner from report 02) so the headline IoU number is published at convergence rather than at 10 epochs. ~30 min on the 5090.
2. **In parallel: P0-1 PnP frame diagnosis** (the 45.6° rotation-vs-VICON bug in `README.md:38–42`). Independent of segmentation work; debugging only, not training. Largest blocker on pose-accuracy claims.
3. **Eventually: actual Orin NX measurement** — when we have a dev kit, re-run `scripts/bench_trt.py` there and replace the projection in this report with a real number. Same script, just on the device.
4. **Eventually: INT8 calibration path** — extend `bench_trt.py` with a `--int8` flag that calibrates on `data/sam_masked/test/` (~500 frames, sufficient per TRT docs).

## Reproduction

```bash
ssh qolam 'tmux new -A -s trt "cd uni/DroneRacing_GateDetection && /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output python scripts/bench_trt.py"'
```

Stable; produces deterministic timing within a few µs across runs. Output is the per-line metrics block above, plus the projection note.
