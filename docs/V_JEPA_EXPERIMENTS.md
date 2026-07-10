# V-JEPA 2-AC Experiments on the YAM Rig — Detailed Plan

Status: proposal for review. Nothing in this repo previously pinned down the
queued V-JEPA experiments, so this document reconstructs the queue as the
canonical set of experiments this rig supports, makes the science explicit,
and flags the methodological traps before we burn robot time. Edit the queue
here; this file is the source of truth going forward.

## 1. What we are testing, in one paragraph

V-JEPA 2-AC (Meta, June 2025) is an action-conditioned latent world model:
a frozen 1B-param ViT-g video encoder plus a ~300M predictor post-trained on
~62 h of unlabeled DROID Franka trajectories. It plans by sampling candidate
end-effector action sequences with the cross-entropy method (CEM) and picking
the sequence whose *predicted future latent* is closest (L1) to the latent of
a *goal image*. Meta reports zero-shot transfer to Franka arms in new labs —
reach essentially always works, grasp and pick-and-place are object-dependent
and substantially weaker — with ~16 s of planning per action. The scientific
question for us is **embodiment transfer**: does a world model trained only on
Franka/DROID video produce a usable energy landscape and usable plans on a YAM
arm it has never seen, and if not, how little YAM data does it take to fix?

Key facts we verified against the released code
(`facebookresearch/vjepa2`, `notebooks/energy_landscape_example.ipynb`):

- Load: `torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")`
  returns `(encoder, predictor)`.
- Observation: monocular RGB, 256×256 center crop, single fixed exocentric
  camera. `tokens_per_frame = (256/16)² = 256`.
- Action: 7D per step — `[dx, dy, dz, drot0, drot1, drot2, gripper]`
  end-effector deltas in the *camera-implied* robot frame (no calibration
  input — the model infers the action frame from pixels).
- Proprio: 7D end-effector pose per frame; the rollout integrates it with
  `compute_new_pose(state, action)`.
- Energy: layer-normed patch latents, `L1(z_pred, z_target)` averaged over
  tokens; planning via `utils/world_model_wrapper.WorldModel.infer_next_action`
  (CEM with `rollout`, `samples`, `topk`, `cem_steps`, `maxnorm` knobs; runs
  even on CPU with tiny budgets).

## 2. What the rig gives us (grounded in this repo)

| Capability | Where | Relevance |
|---|---|---|
| Bimanual 14D joint control over WebSocket | `scripts/yam_control_ws_server.py`, README | Execution backend |
| Cartesian EE deltas via MuJoCo IK, bounded segments | `scripts/yam_cartesian_control.py` (`Kinematics.fk/ik`) | Gives us exactly V-JEPA's 7D EE-delta action interface |
| Fixed top camera (Orbbec color) + depth + two wrist cams | `scripts/multi_camera_ws_server.py` | The exocentric 256px feed and goal images |
| Teleop recording → 7D single-arm episodes (front camera, 10 fps) | `scripts/yam_lerobot_dataset.py` | Offline probing data + fine-tuning data |
| ACT training path (LeRobot) | `yam_lerobot_dataset.py train-act`, `yam_lerobot_policy_server.py` | Behavior-cloning baseline |
| Modal GPU service pattern | `modal_molmoact2_service.py` | Where the 1.3B model actually runs |
| Snapshot/state bundling | `scripts/yam_codex_snapshot.py` | Trial logging |

Compute reality: the robot host is a Mac. A 1B encoder + 300M predictor with
CEM (hundreds of rollouts per action) is not an MPS workload if we want
< ~20 s/action. Plan: **V-JEPA planning service on Modal (A100/A10G, fp16),
same request/response pattern as the MolmoAct2 service; the Mac sends the
current frame + EE pose + goal image, receives one 7D action.** MPS/CPU is
acceptable only for E1 offline probing (no real-time constraint).

## 3. Hypotheses

- **H1 (representation transfer):** V-JEPA 2-AC's energy function, computed on
  YAM top-camera video, assigns lower energy to the true executed action than
  to distractor actions — despite never seeing a YAM arm. This is the cheapest
  falsifiable claim and gates everything else.
- **H2 (action-frame recoverability):** the model's implied action coordinate
  frame on our camera is related to the robot's true EE frame by an
  approximately linear map (Meta reports camera-pose error is ~linear in
  azimuth and correctable by a linear fit to random-action data).
- **H3 (zero-shot control):** with H1+H2 holding, receding-horizon CEM reaches
  goal images for quasi-static single-arm tasks at success rates meaningfully
  above a random/open-loop baseline, with reach ≫ grasp ≫ pick-place.
- **H4 (data efficiency):** a small amount of YAM teleop data (~1–5 h) used to
  post-train the predictor (frozen encoder) closes a large fraction of the
  gap between zero-shot V-JEPA and an ACT policy trained on the same demos —
  i.e., world-model pretraining buys data efficiency on a new embodiment.

## 4. Experiment queue

Ordering is a dependency chain with explicit go/no-go gates so we never debug
closed-loop control on top of an unvalidated representation.

### E0 — Infrastructure & reproduction (no robot)

Stand up the model and reproduce Meta's sanity check before touching our data.

- Load `vjepa2_ac_vit_giant` on Modal; run the energy-landscape notebook on
  Meta's bundled `franka_example_traj.npz`, including the `play_in_reverse`
  control (reversed trajectory must visibly reshape the landscape).
- Benchmark: seconds per CEM action at the paper-scale budget
  (`samples≈100+`, `cem_steps≈10`, `rollout 2`) on the chosen GPU, and at the
  notebook's toy budget. Deliverable: latency table → sets the control loop
  period for E3.
- Wrap as an HTTP `/infer` service mirroring `yam_lerobot_policy_server.py`'s
  shape: request `{frame_jpeg, ee_pose[7], goal_jpeg, mpc_args}` → response
  `{action[7], energy, timings}`.

Exit gate: reproduced landscape matches the notebook qualitatively; service
returns an action end-to-end. If torch.hub weights are unavailable in the
Modal environment, mirror the checkpoint into a Modal volume first.

### E1 — Offline energy probing on YAM video (robot used only for recording)

The core transfer test, done entirely offline so it is cheap to iterate.

Protocol:
1. Record ~30 short teleop clips on the top camera at 4 fps effective
   (subsample our 10 fps recordings), single right arm, simple tabletop scene,
   with synchronized EE poses from `Kinematics.fk` (extend the recorder: today
   episodes store joints only; add `ee_pose` per frame — small code change).
2. For each consecutive frame pair, compute the true EE delta and evaluate the
   energy of (a) the true action, (b) a 5×5×5 grid of translation deltas
   (notebook's `forward_actions`), (c) the reversed-time action.
3. Metrics: rank of the true action within the grid (median rank, top-10%
   rate), energy margin (true vs grid mean), and Spearman correlation between
   energy and Euclidean distance-to-true-action. Repeat with the wrist camera
   as a negative control (expected to fail — DROID is exocentric).

Conditions to sweep (each is a known sensitivity): camera height/azimuth
(2–3 mounts), second arm visible vs parked out of frame, gripper open vs
closed, scene clutter.

Exit gate (go/no-go for E3): median rank of true action in top 20% of the
grid and positive margin in the best camera condition. If this fails
outright, zero-shot control (E3) is dead on arrival; skip to E4
(post-training) and report the negative result — that is itself a real
finding about embodiment transfer.

### E2 — Action-frame calibration

V-JEPA 2-AC's biggest documented failure mode is camera-pose sensitivity: the
model expresses actions in a frame inferred from pixels, which need not match
our robot frame.

Protocol: execute ~50 random bounded EE-delta probes (reuse the probing logic
from `yam_visual_servo_touch.py`, `maxnorm ≤ 0.075 m` matching the model's
clamp), record frame pairs, run single-step CEM inference for each pair
("what action does the model think happened?"), then fit a linear map A from
model-frame actions to commanded robot-frame actions.

Metrics: R² of the fit per axis; condition number of A. Deliverable:
`config/vjepa_action_calibration.json` applied between planner output and
`yam_cartesian_control`-style execution.

Exit gate: R² ≥ 0.6 on translation axes. Below that, the model can't be
steered by linear correction on this camera — move/re-aim the camera to look
more DROID-like (over-the-shoulder, slightly elevated) and repeat once before
declaring failure.

### E3 — Zero-shot goal-image planning (the headline experiment)

Receding-horizon control: capture goal image → loop {grab frame + EE pose →
CEM on Modal → apply calibrated 7D delta via IK → settle} until energy
plateaus or budget exhausted.

Task ladder, single (right) arm, quasi-static, fixed object set:
1. **Reach** — move EE to a marked region (goal image = arm at target).
2. **Push** — displace a box ~15 cm to a target zone.
3. **Grasp** — close gripper on a cup (goal image = gripped cup, subgoal
   image of pre-grasp pose; Meta used subgoals for multi-stage tasks).
4. **Pick-and-place** — cup onto a plate, with 2 subgoal images.

Design rules (pre-registered, to keep us honest):
- N = 10 trials/task/condition minimum, start poses drawn from a fixed 10-pose
  grid used identically for every method. Report success with 95% Wilson
  intervals; per-start-pose pairing for method comparisons.
- Success criteria written down *before* running (e.g. reach: EE within 5 cm
  of target; push: object center in zone; grasp: object lifted 5 cm for 3 s).
  A human scores from the recorded video, blind to method where feasible.
- Every trial logged via a `yam_codex_snapshot.py`-style bundle: goal image,
  all frames, EE poses, planned actions, energies, planning latencies. No
  discarding trials except for rig hardware faults, which are logged.
- Baselines: (a) random-action policy under identical bounds and budget —
  establishes the floor; (b) ACT trained on task teleop demos — establishes
  the "just clone it" ceiling; (c) optionally the existing MolmoAct2 pipeline
  as the VLM-planner comparison. Note the framing: ACT sees task demos,
  V-JEPA sees zero — this is a *data-efficiency* comparison, not a fair
  head-to-head, and the writeup must say so.
- Secondary metrics: final EE/object distance to goal (partial credit),
  number of planner steps, wall-clock per trial, energy-vs-time curves
  (does energy actually decrease? if energy falls but the task fails, the
  energy is exploitable — an important negative signal distinct from
  planner failure).

Safety bounds (all already supported by `yam_cartesian_control.py`): per-step
`maxnorm 0.075 m`, per-joint delta caps, workspace box, gripper clamp
[0.01, 0.59], stop-on-IK-failure.

Expected outcome (from Meta's numbers): reach near-ceiling, grasp/pick-place
possibly much weaker than their Franka results because YAM is out-of-
embodiment. Gripper handling is a known V-JEPA 2-AC weak spot; we map YAM's
normalized gripper to a binary open/close matching DROID conventions rather
than commanding intermediate values.

### E4 — YAM post-training (data efficiency; stretch)

Only if E1–E3 produce a clear picture (either "works, but grasp is weak" or
"representation transfers, control doesn't").

- Convert teleop recordings (target: 1–5 h across the E3 tasks plus play
  data) into DROID-format clips: 4 fps, 256px exocentric frames, EE poses,
  EE-delta actions. The repo's post-training config
  (`configs/train/vitg16/droid-256px-8f.yaml`) is the template; freeze the
  encoder, train the predictor with the same teacher-forcing + rollout loss.
- Evaluate exactly as E1 (energy metrics) and E3 (task success) at data
  scales {0 h (zero-shot), 1 h, 5 h}. Compare against ACT trained on the same
  hours. H4 predicts V-JEPA post-trained on 1 h ≥ ACT on 1 h, with the gap
  closing as data grows.
- Ablation if time allows: predictor fine-tune vs LoRA on predictor only.

### E5 — Cheap side experiments (opportunistic)

- **Encoder probing:** frozen V-JEPA 2 (non-AC) features + linear probe for
  contact / grasp-success prediction from our recordings; compares ViT-L vs
  ViT-g (and V-JEPA 2.1 checkpoints, now released at 384px) at near-zero cost.
- **Depth-assisted goal generation:** we have registered depth; synthesize
  goal images by selecting a target pixel + depth (ties into
  `yam_move_to_depth_pixel.py`) instead of hand-staging goal scenes.

## 5. Methodological flaws we are explicitly ironing out

1. **Unvalidated-transfer trap:** running closed-loop control before checking
   the energy landscape on YAM data conflates "representation doesn't
   transfer" with "planner is misconfigured". Fixed by the E1 gate.
2. **Camera-frame confound:** planner actions live in a pixel-inferred frame.
   Without E2's calibration, failures are uninterpretable. Also: camera must
   be rigidly mounted and *identical* between goal capture and execution —
   any bump invalidates the latent comparison. Use `reset_camera` + a
   fiducial check at session start.
3. **Goal-image leakage/mismatch:** goal images must come from the same
   camera, same lighting, same scene minus the intended change. Staging goals
   by hand-moving objects changes shadows/arm pose; we log goal images and
   review them. For grasp/pick-place, subgoals are required — a single final
   goal image hides the grasp state (Meta's own protocol).
4. **Bimanual distribution shift:** DROID is single-arm. The parked left arm
   in frame is an uncontrolled distractor. E1 measures it; E3 runs with the
   left arm out of the camera frustum unless E1 says it's harmless.
5. **Frame-rate mismatch:** the model was trained at DROID-like temporal
   spacing (~4 fps, 8-frame clips). Feeding 10 fps context or mismatched
   step spacing silently degrades prediction. We fix the observation clock
   at 4 fps equivalents everywhere (recording subsampling + control loop).
6. **Quasi-static assumption:** at ~16 s/action (GPU) the world must not move
   between plan and act. No dynamic scenes; settle-wait after each segment;
   this is stated as a scope limit, not discovered mid-experiment.
7. **Cherry-picking / peeking:** pre-registered success criteria, fixed
   start-pose grid, mandatory trial logging, all trials reported. Success
   rates get confidence intervals; with N=10, only large effects are
   claimable — we say "grasp 2/10 vs reach 9/10" rather than fake precision.
8. **Unfair baseline framing:** ACT-with-demos vs zero-shot V-JEPA is
   data-efficiency, not capability parity. The writeup axis is *success vs
   hours of YAM data*, where zero-shot sits at x=0.
9. **Energy ≠ progress:** CEM can minimize latent L1 while the task fails
   (adversarial/degenerate minima, e.g. arm occluding the object). Logging
   energy-vs-time per trial lets us distinguish "energy is a bad objective
   here" from "optimizer is weak" — different papers-worth of conclusions.
10. **Gripper convention mismatch:** YAM gripper is continuous [0.01, 0.59];
    DROID/model expects its own convention. Binary mapping, verified once in
    E2 by commanding open/close and checking the model's inferred gripper
    action sign.
11. **Proprio units/frames:** `compute_new_pose` integrates 7D poses; we must
    match its position units (meters) and rotation parameterization exactly —
    unit-test our FK→state adapter against Meta's `poses_to_diff` on recorded
    YAM motion before E1 (a wrong rotation convention would silently poison
    every downstream number).

## 6. What "done" looks like

Minimum publishable/demo-able result (hackathon-realistic): E0–E3 for reach
and push, with the E1 energy-transfer table and the E2 calibration plot —
i.e., a quantified answer to "does a DROID-trained world model steer a YAM?"
Stretch: grasp/pick-place rates and the E4 1-hour post-training point.

Deliverables per experiment land in `docs/results/` as a short markdown +
plots, with raw trial bundles under `logs/vjepa-trials/`.

## 7. Open questions to settle before E0 starts

1. GPU budget on Modal (A10G is likely enough for fp16 inference; A100 if we
   want paper-scale CEM budgets at <10 s/action).
2. Which camera is the canonical exocentric view — current top Orbbec color
   mount, or do we re-mount over-the-shoulder to look more DROID-like?
   (Cheap to decide empirically in E1's camera sweep.)
3. How many teleop hours we can realistically record for E4, and on which
   tasks (reuse of the existing bread-toaster recordings depends on adding
   EE poses; new recordings should log them from day one).

## References

- V-JEPA 2 paper: https://arxiv.org/abs/2506.09985
- Code + AC checkpoint: https://github.com/facebookresearch/vjepa2
  (`notebooks/energy_landscape_example.ipynb`, `utils/world_model_wrapper.py`,
  `configs/train/vitg16/droid-256px-8f.yaml`)
