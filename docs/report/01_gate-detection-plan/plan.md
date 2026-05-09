# Plan 01 — Gate Detection: from current state to A2RL April 2026 + AI Grand Prix Nov 2026

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Status** | Draft v0.1 |
| **Audience** | ETHRC Aeronautics perception working group |
| **Companion** | `docs/research/dataset_strategy.pdf` (the *what* and *why*); this file is the *how, when, and done-criteria* |
| **Repo state covered** | as of HEAD on `main` at the time of writing |

---

## TL;DR — first three things to do this week

1. **Diagnose the PnP frame mismatch.** `README.md:38–42` reports 7.4° rotation error vs GT corners but **45.6° vs VICON**, and 52.8 cm vs GT corners but **4.4 m vs VICON**. The 6× ratio between the two is too large to be detector noise — almost certainly a coordinate-frame sign or extrinsics bug downstream of the network. Until this is resolved, *no improvement to the segmentation model can be observed in pose accuracy.* (Owner: ___, ~2–3 days).
2. **Surface the hidden caveat in our headline numbers.** `src/gate_detection/evaluation/benchmark.py:95–98` explicitly uses **ground-truth corners as the QuAdGate prior** during the bench. That means the 82 % / 5.0 px / 39 % / 30.7 px we quote everywhere measure *refinement-given-perfect-prior*, not blind detection. Re-run with a degraded prior (EKF-style noise, or zero prior + dense candidate sweep) and report both numbers side-by-side. (Owner: ___, ~1 day).
3. **Turn on ImageNet-pretrained DeepLabV3+ encoder.** `src/gate_detection/models/deeplabv3plus.py:14` defaults `encoder_weights=None`; `configs/default.yaml` does not override it; `src/gate_detection/inference/predictor.py:175–177` only applies `/255`, no ImageNet mean/std. Set `encoder_weights="imagenet"` and add the standard `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` normalisation in both training and inference paths. Train one epoch and compare TII val IoU. (Owner: ___, ~half day).

These three items are the cheapest path to "the model has *measurably* improved" in the next 7 days.

---

## 1. Context

ETHRC Aeronautics is preparing the `DroneRacing_GateDetection` model for two competitions in 2026:

- **A2RL drone race** (Abu Dhabi, by ASPIRE) — 1.5 m square gate inner / 2.7 m outer in a 100×30 m hall, monocular Jetson Orin NX onboard. The April 2025 winner (MonoRace, [arXiv:2601.15222](https://arxiv.org/abs/2601.15222)) ran a segmentation + QuAdGate + PnP + EKF + MPC pipeline — the *same* shape as ours.
- **The AI Grand Prix** ([theaigrandprix.com](https://www.theaigrandprix.com/)) — Neros standardised drone with ~12 MP wide FPV monocular and ~100 TOPS compute, virtual qualifier May–July 2026, physical qualifier Sep 2026 SoCal, finals Nov 2026 Ohio.

The accompanying research brief (`docs/research/dataset_strategy.pdf`) covers the dataset survey, evaluation tactic, fine-tune protocol, and segmentation-method assessment. **This document is the operational follow-up: phased work items, file:line targets, done-criteria, and decision points.**

---

## 2. Current state — what's there, what's not

Every claim below is grounded in the actual repo at the time of writing. File:line citations are exact.

### 2.1 Models — implemented

- `src/gate_detection/models/gatenet.py:20–77` — narrow custom U-Net, channel divisor `f=4` ⇒ widths `[16, 32, 64, 128, 128]`, **5 sigmoid heads with deep-supervision weights `(4,2,1,1,1)`**. The values flow in via the `SegmentationModelWrapper` constructed at `gatenet.py:77` and via `configs/default.yaml:15` (the `losses/segmentation.py:33` signature default is just a fallback).
- `src/gate_detection/models/deeplabv3plus.py:11–37` — `segmentation_models_pytorch.DeepLabV3Plus` wrapper, ResNet-34 encoder, single sigmoid head, `encoder_weights=None` default at line 14.
- `src/gate_detection/models/sam2.py:18` — `facebook/sam2.1-hiera-large`, used for offline mask refinement only.

### 2.2 Training — implemented

- Entry point: `scripts/train.py` (CLI flags layer over `configs/default.yaml`).
- Loss: per-scale Dice + 2.0×BCE summed with deep-supervision weights, `src/gate_detection/losses/segmentation.py:30–60`. BCE forced fp32 inside `torch.autocast` at line 57 to dodge bf16 `log(0)` overflow.
- Optimiser: AdamW 3e-4, weight decay 1e-4, polynomial-HF schedule (warmup 0.1, power 0.9, lr_end 1e-6), batch 8, 100 epochs, bf16 AMP, channels-last, `torch.compile(mode="max-autotune")` — all in `configs/default.yaml:18–40`.
- Online aug chain: flip → ±15° affine → HSV±10 → motion blur (k 5–15) → σ ≤ 15 noise, in `src/gate_detection/data/transforms.py:91–98`.

### 2.3 Inference + post-process — implemented

- `scripts/inference.py` runs `GatePredictor` on either a list of images or random TII samples, writes a viz grid.
- QuAdGate: `src/gate_detection/postprocessing/quadgate.py:1–329`. LSD lines (params hard-coded at line 63–65: `scale=0.8, sigma_scale=0.8, quant=25.0, ang_th=30°`), pairwise intersection, **4-bit hand-crafted prior bit patterns** (8 entries at lines 35–44), 4-DOF RANSAC with reproj threshold 5 px, sanity-check `|t|≤150 px`.
- PnP: `scripts/benchmark_pnp.py` only — **PnP is not in the deployment loop**. Default gate geometry is `_DEFAULT_GATE_HALF = 0.76 m` (1.52×1.52 m square; the docstring comment at `benchmark_pnp.py:151` says "1.52×1.53 m" but that is wrong — both half-axes default to 0.76 m so the gate is exactly square). Close to but not identical to A2RL's 1.5 m spec.

### 2.4 Evaluation — implemented

- `scripts/evaluate.py` — segmentation IoU at threshold 0.5 (`configs/default.yaml:44`).
- `scripts/benchmark_flight.py` — per-flight detection rate, corner error in pixels, 5 px / 10 px buckets, throughput. **Uses GT corners as QuAdGate priors** (`src/gate_detection/evaluation/benchmark.py:95–98`).
- `scripts/benchmark_pnp.py` — PnP reprojection (vs GT 2-D and detected 2-D), translation error, geodesic rotation error.

### 2.5 Data prep — partially implemented

- `scripts/build_sam_dataset.py` — selects eligible frames (`min_visible≥2` inner corners) from piloted + autonomous flights, fixed-seed `42` train/val/test split, symlinks under `data/sam/{train,val,test}/`.
- `scripts/generate_sam_masks.py` — runs SAM2 on labelled bbox prompts, subtracts inner polygon, writes `<stem>_mask.png` to `data/sam_masked/{train,val,test}/`.
- **Missing despite being documented** in `README.md:93,94,133,143`:
  - `scripts/prepare_labels.py` — supposed to synthesise outer corners (the 17 → 29-col conversion). The synthesis code lives only inline in `scripts/benchmark_pnp.py:79–81`.
  - `scripts/augment_data.py` — supposed to produce a **5× offline augmentation multiplier**. Not in the repo. Only the *online* per-batch chain exists.
  - `src/gate_detection/data/synthesis.py` — referenced in `docs/data.md` and the research doc but does not exist on disk.

### 2.6 Reported metrics — and the caveat

| Quantity | Value | `README.md` line |
|---|---|---|
| Detection rate, TII (`flight-01p-fixed`) | **82 %** | 32 |
| Detection rate, real recorded data | **39 %** | 48 |
| Mean reprojection error, TII | **5.0 px** | 35 |
| Mean reprojection error, real | **30.7 px** | 44 |
| Translation error vs GT corners | 52.8 cm | 37 |
| **Translation error vs VICON** | **4.4 m** | 39 |
| Rotation error vs GT corners | 7.4° | 38 |
| **Rotation error vs VICON** | **45.6°** | 40 |

**Critical caveat:** the 82 % / 5.0 px / 39 % / 30.7 px numbers come out of `benchmark.py` which uses **GT corners as QuAdGate priors** — see `src/gate_detection/evaluation/benchmark.py:95–98`:

> *Use GT corners as the QuAdGate prior. This is a benchmarking convenience — at deployment the prior would come from the EKF / corner refinement quality without confounding from prior drift.*

That means our publicly-quoted detection rate is *"given a perfect prior, can the refinement step recover the corners?"* — not blind detection-from-scratch. It is still a meaningful number (it isolates QuAdGate refinement quality) but anyone using it to estimate field performance is reading it wrong.

### 2.7 Explicit "Not Done" — `README.md:50–55`

1. ROS2 integration (no node, publisher, or launch files).
2. Real-data fine-tuning (no script, no domain-adaptation pipeline).
3. PnP coordinate-frame alignment (VICON ↔ camera world frame transform).
4. Online / real-time inference (no latency benchmarking, no embedded export).

### 2.8 Missing infrastructure (not in any "Not Done" list)

- **No tests anywhere** — no `tests/` directory, no `pytest`, no `.github/workflows/`, nothing CI-shaped.
- **No resume-from-checkpoint** in `scripts/train.py`. A run that crashes at epoch 89 of 100 starts over.
- **No logging beyond `print`** — no TensorBoard, no W&B, no JSON metric dumps. Run-to-run comparison is by hand.
- **Two README-documented scripts missing on disk** (§ 2.5).
- **No ImageNet normalisation** in `src/gate_detection/inference/predictor.py:175–177` — pre-processing is `BGR→RGB` then `/255` only.
- **HSV jitter ±10 on hue** (`transforms.py:52`, `h_range=10.0` default) is way narrower than any realistic competition-venue colour shift; the gate/lighting hue could swing by ±60°.

---

## 3. Assumptions and out-of-scope

This plan assumes:

- The hardware target for A2RL is **Jetson Orin NX** (per MonoRace [arXiv:2601.15222](https://arxiv.org/abs/2601.15222), which reports the A2RL drone as a 966 g racer with NVIDIA Jetson Orin NX as its main processing unit). If the team migrates to **Jetson Thor** before April 2026, the deployment phase changes (Isaac ROS 4.x is Thor-only).
- The repo's gate-segmentation + QuAdGate + PnP pipeline shape is kept (per the research-doc verdict that the *method* is right, only the *generalisation* is failing). A pivot to pure YOLOv8-Pose-keypoint regression (à la Azhari et al. 2025, [arXiv:2512.20475](https://arxiv.org/abs/2512.20475)) is **deliberately out of scope** here — but is the obvious fall-back if this plan stalls.
- The TII v3.0.0 dataset stays the primary training corpus; the AI Grand Prix qualifier simulator (released May 2026) becomes a secondary corpus once available.
- Time horizon: the plan is sized to **complete P0 and P1 by end of next month**; P2 by Feb 2026; P3 well before A2RL practice slots in April 2026; P4 at each competition.

Out of scope: end-to-end policy learning (Swift/Nature 2023-style RL), full-stack VIO, controller tuning, mechanical drone work.

---

## 4. The plan

Each item has a **priority within phase**, **time estimate** (S = ≤1 day, M = 2–5 days, L = ≥1 week), **owner placeholder**, **done-criterion**, and **touch-points** in the repo.

### Phase P0 — Fix the load-bearing bugs (this week)

Until P0 is done, *no model improvement can be measured in pose accuracy* and *the headline numbers are misleading*. P0 is the gate to everything else.

#### P0-1 · Diagnose and fix the PnP frame mismatch — **L** — Owner: ___

- **Why**: `README.md:38–42`. Rotation error vs VICON is **6.2× larger** than vs GT corners (45.6° vs 7.4°), translation error is **8.3× larger** (4.4 m vs 52.8 cm). That ratio is consistent with a coordinate-frame sign error (e.g. `R_vicon = R_pnp · diag(1, -1, -1)` or similar), not detector noise.
- **Touch-points**: `scripts/benchmark_pnp.py`, `src/gate_detection/evaluation/pnp.py`, the calibration files referenced via `--calib`.
- **What to do**:
  1. Print every coordinate-frame transform on a single frame: `R_camera_world` (PnP output), `R_drone_world` (VICON pose at sync time), `R_camera_drone` (extrinsic). Compare against the canonical TII frame convention (z-down body or z-up body — quote the doc).
  2. On a frame where `flight-01p-fixed/000.jpg` GT-corner-based PnP gives `t = 52.8 cm` vs GT, work backwards: which frame is the residual *4.4 m* error in?
  3. Likely fix: a sign flip in `R_camera_drone` or a missed `cv2.Rodrigues` orientation between OpenCV (right-handed, z-forward) and the VICON frame (typically z-up).
- **Done = on `flight-01p-fixed`, rotation error vs VICON ≤ 10°, translation error vs VICON ≤ 1 m**, both averaged over 360 frames at stride 30.

#### P0-2 · Re-run and re-publish the benchmarks with a *realistic* prior — **M** — Owner: ___

- **Why**: `src/gate_detection/evaluation/benchmark.py:95–98` uses GT corners as the QuAdGate prior. Field performance won't have that. We need numbers that bracket reality.
- **Touch-points**: `src/gate_detection/evaluation/benchmark.py`, `scripts/benchmark_flight.py`.
- **What to do**:
  1. Add a CLI flag `--prior-mode {gt, noisy, ekf, none}` to `scripts/benchmark_flight.py`.
  2. `noisy`: add a configurable σ jitter to GT priors (default σ = 30 px).
  3. `ekf`: use the previous frame's detected corners as the prior for the next frame, with re-init from GT every `N` frames.
  4. `none`: pass zero prior; force QuAdGate to do candidate-only matching (this exercise's main purpose is to flag whether QuAdGate even *can* run blind).
  5. Re-quote the headline numbers in `README.md` and `docs/research/dataset_strategy.tex` with **all four** prior modes side by side.
- **Done = `README.md` § Known Issues lists detection rate and reprojection for {gt-prior, noisy-σ-30-prior, ekf-prior, no-prior} on both `flight-01p-fixed` and the real-recorded set**.

#### P0-3 · ImageNet weights + ImageNet normalisation — **S** — Owner: ___

- **Why**: §4.4 of the research doc; `deeplabv3plus.py:14`; `predictor.py:175–177`.
- **Touch-points**:
  - Add `model.encoder_weights: imagenet` to `configs/default.yaml`.
  - Plumb through `src/gate_detection/models/factory.py` (currently doesn't pass `encoder_weights` from config to the builder).
  - Add ImageNet mean/std to `predictor._preprocess_bgr` and to the training transforms (`src/gate_detection/data/transforms.py`) so train/inference match.
- **Done = TII val IoU reported for {current baseline, baseline + ImageNet-weights-only, baseline + ImageNet-weights + ImageNet-norm}, each retrained for ≥3 epochs at fixed seed (one epoch is too noisy to discriminate)**. If the change lifts val IoU by ≥1 pp without regressing piloted-vs-autonomous balance, ship it; otherwise log and keep off.

#### P0-4 · Locate or rewrite the missing data scripts — **M** — Owner: ___

- **Why**: `scripts/prepare_labels.py` is referenced on `README.md` lines 93, 133, 134 (3×); `scripts/augment_data.py` on lines 94 and 143 (2×); `src/gate_detection/data/synthesis.py` is referenced in `docs/data.md` (line 74). Six references total across docs, none of those three files exists on disk.
- **What to do**: either (a) recover them from a stash branch / a teammate's local checkout (likely), or (b) reconstruct as small wrappers around `parse_pixel_corners` (already in `src/gate_detection/data/labels.py`) for the prepare-labels case; for `augment_data.py`, write a minimal driver that calls `train_augment` from `transforms.py:91` over a glob of inputs, writing N augmented copies per source.
- **Done**: both scripts exist and `python scripts/prepare_labels.py --help` / `python scripts/augment_data.py --help` print a sane usage. Unit-smoke test: run on one flight, confirm output files appear with the documented schema.

---

### Phase P1 — Strengthen the model (next 2–3 weeks)

P1 makes the model *demonstrably* better on TII *and* on the real-recorded set we built in P0-2. Each item is a separate, comparable training run.

#### P1-1 · Add a parallel keypoint-heatmap head — **L** — Owner: ___

- **Why**: `azhari2025drift` (A2RL podium) reports **0.971 keypoint mAP** at 62 Hz on Orin NX; pure mask-only is leaving information on the floor.
- **Touch-points**:
  - New file `src/gate_detection/models/heads/keypoint.py`: 1×1 conv → 4-channel heatmap (Gaussian peak per inner corner) at `H/4 × W/4`.
  - `src/gate_detection/models/{gatenet,deeplabv3plus}.py`: branch the decoder feature into the existing seg head AND the new keypoint head.
  - `src/gate_detection/losses/keypoint.py`: focal MSE on Gaussian heatmaps.
  - Multi-task weighting: start with **uncertainty weighting** (Kendall et al. 2018, [arXiv:1705.07115](https://arxiv.org/abs/1705.07115); each task has a learned `log_sigma`), fall back to fixed `1.0×L_seg + 1.0×L_kp` if the learned weights collapse.
- **Done = TII val keypoint mAP@5px reported, AND TII val seg IoU is no worse than P0-3 baseline**. The keypoint output is fed into QuAdGate as a higher-quality prior than the EKF prior currently planned.

#### P1-2 · Build the real-footage evaluation set ("Tier 2") — **M** — Owner: ___

- **Why**: research-doc §5; the 39 % real-recorded number is currently an opaque single-flight reading. Tier 2 needs ≥300 hand-labelled frames covering lighting + pose + distance variety.
- **Touch-points**: `scripts/viewer.py` (already exists; supports accept/reject/edit corners), a new `scripts/build_real_eval_set.py` that walks team-recorded flight folders and dispatches into viewer with the current best checkpoint pre-filling pseudo-labels.
- **Done = `data/real_eval/` with ≥300 frames, label files in 17-col TII format, an `eval_real_flight.py` that prints detection rate / mean reprojection / 5 px / 10 px buckets**. Mirrors the existing `benchmark_flight.py` schema so numbers compare directly.

#### P1-3 · Pencil-filter as a 4th input channel — **M** — Owner: ___

- **Why**: PencilNet ([arXiv:2207.14131](https://arxiv.org/abs/2207.14131)) Algorithm 1 — `G ← grayscale; P ← dilate(G, ellipse); P ← 255·G/P` — is the cheapest known intervention with peer-reviewed real-flight sim-to-real evidence. Implement once, run as ablation on Tier 1 + Tier 2.
- **Touch-points**:
  - `src/gate_detection/data/transforms.py`: new `pencil_filter(img_rgb) -> np.ndarray (H, W)`.
  - `src/gate_detection/data/dataset.py`: optional 4-channel mode (concat RGB + pencil) gated on `cfg["data"]["use_pencil"]`.
  - `src/gate_detection/models/{gatenet,deeplabv3plus}.py`: first-conv `in_channels=4` when 4-channel mode active.
- **Done = ablation table in `docs/report/02_pencil-ablation/findings.md` with {3-ch baseline, 4-ch pencil} × {Tier 1, Tier 2} = 4 cells**. If pencil gives ≥+5 pp on Tier 2 with ≤−2 pp on Tier 1, ship; else file as inactive.

#### P1-4 · Resume-from-checkpoint, TensorBoard logging, basic tests — **M** — Owner: ___

- **Why**: §2.8 missing infrastructure. None of the above is research; all of it is engineering hygiene that pays off compoundingly.
- **Touch-points**:
  - `scripts/train.py`: `--resume` flag → load weights, optimizer, scheduler, RNG state from latest snapshot.
  - `src/gate_detection/training/trainer.py`: add a `tensorboardX` writer (or `torch.utils.tensorboard`); log per-step train loss + per-epoch IoU.
  - New `tests/` with: (a) model-build smoke (`build_gatenet()` + `build_deeplabv3plus()` produce models that accept a `(1,3,384,384)` tensor); (b) loss-compute smoke (run `gate_loss` on dummy outputs); (c) postproc smoke (run QuAdGate on a synthetic ring mask, check it returns 8 points within 5 px of the ground truth); (d) discovery-determinism (assert `flight_dirs("train", DATA)` returns the same flights on two calls under `random.seed(42)`).
  - `.github/workflows/test.yml` runs the test suite on every push.
- **Done = `pytest tests/` passes, CI green on a PR, `--resume` round-trips a checkpoint without measurable metric drift on TII val**.

---

### Phase P2 — Sim-to-real bridge (next month)

P2 is the synthetic-data work. It explicitly does not block P3 deployment — if the simulator integration takes longer than scoped, ship P0+P1 onto hardware first.

#### P2-1 · Render synthetic A2RL-spec gates in AirSim NeurIPS-2019 — **L** — Owner: ___

- **Why**: research-doc §5; the only public source we have for "indoor flight halls with arbitrary gate geometry". A2RL spec is 1.5 m / 2.7 m / square (per MonoRace [arXiv:2601.15222](https://arxiv.org/abs/2601.15222)).
- **Touch-points**:
  - Fork the `microsoft/AirSim-NeurIPS2019-Drone-Racing` repo (note: archived; build from source on Linux/x86 with UE 4.27).
  - Author a new `scripts/synth_render_a2rl.py` that spawns 1.5 m gates at random poses, samples a camera trajectory at A2RL camera intrinsics (155°×115° FoV from MonoRace Table I), saves images + ground-truth corners using `simSetSegmentationObjectID`.
  - Output schema matches TII: `data/synthetic_a2rl/{train,val,test}/flight-XX/camera_*/...`, `label_*/...`.
- **Done = ≥10k frames generated, ablation cell {TII-only, TII + synth} on Tier 1 + Tier 2**. If TII+synth matches or beats TII-only on Tier 2, ship synth into the training stack.
- **Risk**: AirSim NeurIPS-2019 is archived (research-doc §7); if it doesn't build, fall back to **Isaac Sim** (free since Aug 2024) or **Unity-based ETHRC sim** if the team has one.

#### P2-2 · (Optional) NeRF / 3DGS reconstruction of the local flying space — **L** — Owner: ___

- **Why**: FalconGym ([arXiv:2503.02198](https://arxiv.org/abs/2503.02198)) reports **95.8 % real-flight success** on 38 cm-radius gates after training only in NeRF. If we have ~1 hour of slow-fly footage of the ETHZ practice space, this is doable with `nerfstudio` or `gsplat` in a day.
- **Touch-points**: external — Nerfstudio CLI; pre-existing flight footage; output is a render-on-demand simulator at the actual training space.
- **Done = a render server that accepts a 6-DoF pose query and returns an A2RL-spec gate composited at that pose; ablation: TII + NeRF-renders vs TII-only**.
- **Decision gate**: only start P2-2 if P2-1 hits its done-criterion. NeRF is the higher-fidelity but riskier path.

#### P2-3 · (Optional) SAM2 → student distillation — **L** — Owner: ___

- **Why**: research-doc §4.4; MobileSAM ([arXiv:2306.14289](https://arxiv.org/abs/2306.14289)) shows that a ~10 ms student SAM is feasible. Our codebase already runs SAM2 offline (`scripts/generate_sam_masks.py`); a student would let us skip the SAM-prepass entirely and possibly run mask refinement at inference time.
- **Touch-points**: new `scripts/distill_sam_student.py`, new `src/gate_detection/models/sam_student.py` (probably a small Hiera-tiny variant or a U-Net head off our DeepLabV3+ backbone with SAM2's output as the supervision signal).
- **Done = student inference latency ≤ 10 ms on Orin NX (TensorRT FP16), and student ring-mask IoU ≥ 0.90 of SAM2's mask on TII val**.
- **Decision gate**: nice-to-have. Drop without regret if P3 deployment is at risk.

---

### Phase P3 — Onboard deployment (before A2RL practice in April 2026)

P3 is purely engineering. The gate is "model runs at ≥30 fps on Orin NX in a ROS 2 graph that the controller can subscribe to".

#### P3-1 · ONNX export + TensorRT engine — **M** — Owner: ___

- **Touch-points**:
  - New `scripts/export_onnx.py`: `torch.onnx.export(model, dummy_384, ..., opset=17, dynamic_axes=None)`. Verify with `onnxruntime` that PyTorch and ONNX agree to ≤ 1e-4 on a sample TII frame.
  - Generate engine: `trtexec --onnx=gatenet.onnx --fp16 --useCudaGraph --saveEngine=gatenet.engine --shapes=input:1x3x384x384`.
  - Repeat for INT8 with calibration data drawn from TII train.
- **Done = `gatenet.engine` (fp16) and `gatenet_int8.engine` exist; `trtexec --loadEngine=...` reports a stable mean latency on Orin NX**. Target: **≥ 30 fps fp16, ≥ 60 fps int8**. If model is too heavy, fall back to GateNet-f4 which is much smaller.

#### P3-2 · Isaac ROS integration (`isaac_ros_unet`) — **M** — Owner: ___

- **Why**: prior turn's research; `isaac_ros_unet` accepts our exact spec (NCHW, sigmoid, 384-even, single 32-bit-float output). NITROS gets us zero-copy GPU buffers between camera → seg → downstream.
- **Pin**: Isaac ROS **3.2 Update 10** (Jul 2025), JetPack 6.1, ROS 2 Humble. **Not** the 4.x line — that's Thor-only and our hardware is Orin NX.
- **Touch-points**:
  - New `ros2_ws/src/gate_detection_ros/` ROS 2 package wrapping `isaac_ros_unet` with our engine.
  - YAML config pointing at `gatenet.engine`, `network_output_type: sigmoid`, threshold 0.5.
  - A small relay node that takes the segmentation output, runs QuAdGate (CPU, OpenCV LSD) and PnP, publishes a `geometry_msgs/PoseStamped`.
- **Done = `ros2 launch gate_detection_ros gate_pipeline.launch.py` produces pose messages at ≥ 30 Hz on the Orin NX bench rig**.

#### P3-3 · Latency budgeting + bench drone field test — **M** — Owner: ___

- **Done = a single-flight log on the practice drone where the segmentation node produces masks for ≥ 95 % of camera frames at ≥ 30 fps, no GPU memory leaks across 5-minute runs, no single-frame latency spike > 50 ms**.

---

### Phase P4 — Competition-time adaptation (at each event)

This is the same protocol described in research-doc §6, operationalised.

#### P4-1 · On-site data collection (5–10 min FPV per venue) — **S** — Owner: ___

- 3+ spatial angles (head-on / 30–45° oblique / sharp oblique).
- ≥ 2 lighting conditions (with stage lights, without).
- 30–50 background frames with no gates.
- Static photographs of the gate corners with a tape measure visible — confirms published dimensions.

#### P4-2 · Pseudo-label → manual refine — **S** — Owner: ___

- Run current best model on collected frames.
- `scripts/viewer.py` accept / reject / edit (already implemented).
- Stop at 200–300 frames.

#### P4-3 · Fine-tune with rehearsal — **M** — Owner: ___

- Freeze encoder, train decoder + last 2 conv blocks at 0.1× LR.
- 1:1 mini-batch mix of competition frames and TII rehearsal samples.
- Stop early on the **joint criterion**: best on competition split AND no regression on TII val. If that joint criterion can't be met within 10 epochs, *more data, not more epochs* — go back to P4-1.

#### P4-4 · Three-tier validation gate before deployment — **S** — Owner: ___

A fine-tuned checkpoint replaces the deployed one **only if all three pass**:

1. TII val: no > 2 pp IoU regression on either autonomous-only or piloted-only.
2. Real-footage benchmark (Tier 2 from P1-2): detection rate ≥ 60 %, reprojection within 2× the TII number.
3. On-site smoke test: no novel false-positive class (e.g. a wall structure consistently segmented as gate) beyond the offline benchmark.

A regression on *any* tier is an automatic abort to the previous deployed checkpoint.

---

## 5. Decision points / kill criteria

| Trigger | Then |
|---|---|
| P0-1 stalls >1 week without a clear hypothesis | Bring in a second pair of eyes; pause downstream pose work and treat detection-rate metrics as the only meaningful signal until pose is trustworthy. |
| P1-1 keypoint head regresses TII val IoU > 2 pp | Drop multi-task; keep mask-only. |
| P1-3 pencil filter does not lift Tier 2 by ≥ 5 pp | File as inactive (do not enable in `configs/default.yaml`). |
| P2-1 AirSim NeurIPS-2019 fails to build on team hardware | Pivot to Isaac Sim before sinking >2 days. |
| P3-1 ONNX export accuracy diff > 1e-3 vs PyTorch | Investigate ops (`align_corners`, `interpolate` modes); do not ship engine. |
| P3-2 Orin NX inference falls below 20 fps even with INT8 | Switch deployed model from DeepLabV3+ to GateNet-f4. |
| Within 4 weeks of A2RL, the real-footage benchmark detection rate is still < 50 % | Treat as a "we will not have an autonomous flight ready" signal; downscope to a teleoperated entry or skip the season. |

---

## 6. Risks and open questions

1. **PnP frame mismatch is not a one-line fix.** The 45.6° error is consistent with a sign flip but can also be incremental (small-angle bias in camera-drone extrinsic + accumulating scale error in VICON sync). Worst case, P0-1 needs a full extrinsic calibration redo.
2. **A2RL gate appearance is unconfirmed.** Geometry is documented; colour, surface finish, reflectivity are not. Plan for a worst-case white powder-coated frame against grey concrete and confirm against ASPIRE technical bulletins as soon as 2026 specs publish.
3. **AI Grand Prix Round-1 simulator API is unreleased** (May 2026). Camera intrinsics, message types, and the Python interface are unknown. Until the simulator drops, anything AIGP-specific is speculative.
4. **No public AlphaPilot dataset** despite multiple writeups citing one. Plan must not depend on access; the synthesis pipeline + on-site collection have to be sufficient on their own.
5. **`scripts/augment_data.py` and `scripts/prepare_labels.py` referenced but absent** — if no teammate has them, P0-4 becomes a 1-week reconstruction job rather than a half-day recovery.

---

## 7. References

In-line citations follow the published research brief (`docs/research/refs.bib`). The minimum set load-bearing for *this plan*:

- Bahnam et al. 2026, MonoRace — A2RL April 2025 winner, [arXiv:2601.15222](https://arxiv.org/abs/2601.15222). Pipeline shape, drone hardware, gate dimensions.
- Azhari et al. 2025, "Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing" — A2RL 2025 podium, [arXiv:2512.20475](https://arxiv.org/abs/2512.20475). YOLOv8-Pose 0.971 keypoint mAP at 62 Hz on Orin NX (Table 3) — basis for the keypoint-head recommendation.
- Pham et al. 2022, PencilNet — [arXiv:2207.14131](https://arxiv.org/abs/2207.14131). Algorithm 1 = the morphological filter that P1-3 implements.
- Qiao et al. 2024, "Continual Learning for Robust Gate Detection under Dynamic Lighting" — [arXiv:2405.01054](https://arxiv.org/abs/2405.01054). Real-flight illumination-robustness evidence.
- Kendall et al. 2018, "Multi-Task Learning Using Uncertainty to Weigh Losses" — [arXiv:1705.07115](https://arxiv.org/abs/1705.07115). Multi-task weighting scheme used in P1-1.
- Miao et al. 2025, FalconGym — [arXiv:2503.02198](https://arxiv.org/abs/2503.02198). 95.8 % real-flight success on 38 cm gates after training only in NeRF — basis for P2-2.
- Zhang et al. 2023, MobileSAM — [arXiv:2306.14289](https://arxiv.org/abs/2306.14289). 10 ms inference, basis for P2-3 distillation.
- Isaac ROS 3.2 Update 10 release on GitHub — [v3.2-10 tag](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation/releases/tag/v3.2-10), the Orin-NX-supporting line we pin to in P3-2.

---

## 8. How this document was produced and how reliable it is

- Inventory section (§ 2) was built by reading the actual files in this repo at HEAD; every file:line citation has been opened and verified.
- Numerical claims about the model (82 %, 39 %, 5.0 px, 30.7 px, 45.6°, 4.4 m) trace to specific lines in `README.md` § Known Issues.
- Citations in § 7 were each fetched via WebFetch (arxiv abstracts and, for the four most-load-bearing papers, the full PDFs); claims attached to them have been re-verified against the source rather than recalled.
- A reviewer-agent cross-validation pass was run on this document before commit; defects it found were fixed in-place.
- **What I cannot vouch for**: I have not run the trained model on real images (no checkpoint in `checkpoints/`); the latency estimates in P3 are extrapolated from comparable architectures (Azhari YOLOv8s, PeopleSemSegNet) and need a real `trtexec` measurement to be trusted; the AirSim NeurIPS-2019 build risk in P2-1 is genuinely unknown until someone tries.

The plan deliberately puts the cheapest-to-falsify items first (P0) and the most-time-investing items behind decision gates (P2 / P3 fall-backs). If only P0 happens, the model is *measurably* in better shape and the headline numbers are *honest*. Everything beyond P0 is upside.
