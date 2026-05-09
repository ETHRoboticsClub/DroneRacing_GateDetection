# Report 04 — Hypothesis on the 45.6° PnP-vs-VICON rotation error (plan P0-1)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Status** | **Hypothesis only — not verified.** Reproduction needs `flight-01p-fixed` data which is not in the TII public release; per `README.md:53` it lives on Google Drive. |
| **Implements (partially)** | `docs/report/01_gate-detection-plan/plan.md` § P0-1 |

## TL;DR

The 45.6° rotation error vs VICON (and 4.4 m translation error) reported in `README.md:38–42` for `flight-01p-fixed` is almost certainly a **calibration-routing bug**, not a generic coordinate-frame sign flip.

`flight-01p-fixed` does not match any case in `_track_type_key()` at `src/gate_detection/evaluation/pnp.py:44–61` — the function silently falls through to `return "ellipse"`. That makes the loader pick `calib_ap-ellipse-lemniscate.json` for intrinsics and the `"ellipse"` rotation key from `drone_to_camera.json` for the extrinsic. **`drone_to_camera.json` has a separate `"piloted"` rotation key that nothing in the routing logic ever selects** — almost certainly the right one for `-fixed`.

The fix is one routing entry plus an audible warning when the routing falls back to its default. Verification needs the actual `flight-01p-fixed` flight data; both pieces are listed at the end of this report.

## Evidence chain

### 1. The "vs GT corners" path is internally consistent

`scripts/benchmark_pnp.py:356–359` computes `t_err` and `r_err` between two `solvePnP` outputs:

- `pnp_det.tvec/rvec` — solvePnP given the network's *detected* 2D corners and the world-frame 3D markers.
- `pnp_gt.tvec/rvec` — solvePnP given the *ground-truth-labelled* 2D corners and the same world-frame 3D markers.

Both go through identical calibration. The 7.4° / 52.8 cm gap is just the cost of imperfect detected corners. **Internally consistent — *not* the bug.**

### 2. The "vs VICON" path uses the calibration directly

`scripts/benchmark_pnp.py:374–375` compares `pnp_gt.rvec` (the camera pose computed by solvePnP using *ground-truth* 2D corners) against `v_rvec` returned by `vicon_camera_pose()` at `pnp.py:318–352`. The latter is built from VICON drone pose × camera-in-body extrinsic:

```
R_w2b = drone_rot[0..8]                                 # from VICON CSV
R_w2c = R_cam_body.T @ R_w2b                            # pnp.py:349
tvec  = -(R_cam_body.T @ (R_w2b @ d_pos + t_cam_body))  # pnp.py:350
```

Where `R_cam_body` is the 3×3 rotation built from the quaternion in `drone_to_camera.json` at `pnp.py:114`. **If `R_cam_body` is the wrong quaternion, the VICON path produces a camera-in-world pose that disagrees with the PnP-derived one — exactly the symptom in the README.**

### 3. `flight-01p-fixed` falls through every case in `_track_type_key`

`src/gate_detection/evaluation/pnp.py:44–61`:

```python
def _track_type_key(flight_name: str) -> str:
    name = flight_name.lower()
    if "ellipse" in name:                # → ellipse        (NO match for fixed)
        return "ellipse"
    if "lemniscate" in name:             # → lemniscate     (NO match)
        return "lemniscate"
    if "trackratm" in name or ...:       # → trackratm-p/a  (NO match)
        ...
    return "ellipse"                     # ← silent default fallback
```

`"fixed"` is not handled. Result: `_track_type_key("flight-01p-fixed") == "ellipse"`.

Confirmed by `grep -rni "fixed\|01p-fixed"` across the entire codebase — only README.md and plan.md mention the flight, no code path handles it.

### 4. The wrong rotation key is then picked

With `key == "ellipse"`, lines 98–107 set `qkey = "ellipse"`. Lines 110–113 read `R_cam_body` from `drone_to_camera.json` under the `"ellipse"` key.

`drone_to_camera.json` actually contains **four** rotation keys:

```
{ "rotation": { "lemniscate": ..., "ellipse": ..., "trackRATM": ..., "piloted": ... } }
```

The `"piloted"` key — likely the right one for `flight-01p-fixed` — is **never selected** by the routing logic. Lines 95–96 even define a `track_family = "piloted" if key.endswith("-p") or key in ("ellipse", "lemniscate") else "lemniscate"` variable that is **assigned and then ignored** (it is overwritten by the `qkey = ...` block immediately below). Whether that's intentional or dead code, the net effect is the same: `qkey` is always one of `{ellipse, lemniscate, trackRATM, piloted}` *based purely on the matched key string*, and "piloted" can only be reached when the key is unrecognised — but the function returns "ellipse" before that can happen.

### 5. The error magnitudes are consistent with a wrong extrinsic rotation

`R_cam_body` mediates *both* the rotation and translation in `vicon_camera_pose`:

```
R_w2c = R_cam_body.T @ R_w2b
tvec  = -(R_cam_body.T @ (R_w2b @ d_pos + t_cam_body))
```

A wrong `R_cam_body` therefore corrupts both rotation and translation simultaneously, in the magnitudes observed:

- A rotation error `δ` between the actual camera-in-body and the (wrong) "ellipse" extrinsic appears as a `δ`-degree error in `r_err_vicon`. **45.6° is plausible for a different mounting orientation between the ellipse and the fixed rig.**
- The translation chain includes `R_cam_body.T @ t_cam_body`. If the rotation is off by 45° and `|t_cam_body|` is small (say ~10 cm), the offset of the camera-position-in-world is small. But `R_w2c = R_cam_body.T @ R_w2b` rotated incorrectly compounds with the *world-distance* from the camera to the gate (~5–10 m at A2RL distances), giving tvec errors in the metre range. **4.4 m matches the order of magnitude.**

The internal-consistency check (`vs GT corners` errors are small) further rules out anything wrong with `R_w2b` or the world 3D markers — those are shared between the two error paths.

## Proposed fix (in order of size)

### Patch 1 (this commit) — make the fallback non-silent

CLAUDE.md §6 is explicit: *"defaulted config values without announcing them are bugs"*. The current `_track_type_key` silently picks `"ellipse"` for any unrecognised name, which means **anyone running benchmark_pnp on a new flight gets quietly miscalibrated results unless they happen to inspect the code**. The minimal fix is to print a `[WARN]` whenever the default kicks in, so the issue surfaces in any future run — not just for `-fixed` but for any flight type that gets added later.

Applied in this commit. Sample output under fallback:

```
[WARN] _track_type_key('flight-01p-fixed'): no pattern matched
       (expected one of: ellipse | lemniscate | trackratm-* );
       defaulting to 'ellipse'. If the flight is mounted differently,
       its calibration may be wrong — pass --calib-key explicitly.
```

That alone would have surfaced the bug at the first run on the `-fixed` flight.

### Patch 2 (separate change, after the data lands) — add CLI overrides

Add `--calib-key {ellipse,lemniscate,trackratm-p,trackratm-a}` and `--rot-key {ellipse,lemniscate,trackRATM,piloted}` flags to `scripts/benchmark_pnp.py`. The latter is the load-bearing one for this bug — it lets us try `--rot-key piloted` on `flight-01p-fixed` and re-measure.

### Patch 3 (after a clear verdict) — fix the routing

Once we have data and have confirmed which extrinsic key is right for `-fixed`, add the explicit case to `_track_type_key`:

```python
if "fixed" in name:
    return "piloted"
```

(or whatever string the calibration map ends up using). Plus a corresponding entry in `_CALIB_MAP` if the intrinsics differ.

## Verification plan (when `flight-01p-fixed` data lands)

1. **Get the data.** The README says it's on Google Drive. Drop it under `data/piloted/flight-01p-fixed/` on qolam.
2. **Confirm the fallback warning.** Run `python scripts/benchmark_pnp.py data/piloted/flight-01p-fixed --calib calibration/`. The `[WARN]` from Patch 1 should fire on every frame. If it does not, the fallback isn't the issue; investigate further.
3. **Try `--rot-key piloted` (after Patch 2 lands).** Re-run with `--rot-key piloted`. Expected outcome: rotation error vs VICON drops from 45.6° to <10°, translation error vs VICON drops from 4.4 m to <1 m.
4. **Lock in the fix.** Whichever key produces the smallest VICON-vs-PnP error becomes the canonical mapping in `_track_type_key` (Patch 3).
5. **Update `README.md:38–42`** with the new error numbers; the original ones are misleading until the calibration routing is fixed.

## What this report does not claim

- This is a **hypothesis**, not a verified diagnosis. I have no `flight-01p-fixed` data on hand. The error pattern is consistent with the calibration mismatch hypothesis; nothing rules out alternatives.
- Alternative explanations I considered, in decreasing likelihood:
  - **Different VICON convention** (`drone_rot[0..8]` stored body→world for `-fixed` instead of world→body) — possible but less likely; would also break `vicon_gate_selection`.
  - **VICON time-sync offset** in `-fixed` flight log — would explain large translation errors at speed but not a clean 45° rotation offset.
  - **Different CSV schema** in `-fixed` flight — the code reads `drone_x/y/z` and `drone_rot[0..8]` columns; if `-fixed` uses different names, the code raises and we'd already know.
- The 7.4° / 52.8 cm "vs GT corners" baseline is a statement about *detector quality given perfect corners*, **not** about the underlying calibration bug. That number stays where it is regardless of how P0-1 ends.

## Reproduction (when data is available)

```bash
ssh qolam 'cd uni/DroneRacing_GateDetection && \
  /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
    python scripts/benchmark_pnp.py data/piloted/flight-01p-fixed \
      --calib calibration/ 2>&1 | tee logs/pnp_fixed_$(date +%Y%m%d_%H%M%S).log'
```

The `[WARN]` from Patch 1 will fire on every call to `_track_type_key`. Patch 2/3 follow.
