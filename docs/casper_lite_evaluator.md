# CASPER-lite Evaluator

This evaluator is the first offline step toward a CASPER-style shared-autonomy
baseline. It evaluates VLM-style candidate intent selection before adding an
online ROS node.

## Scope

This is not a full CASPER reproduction. It keeps the existing candidate set and
skill library, then evaluates whether a CASPER-style model can choose the
correct candidate intent from:

- task instruction
- task type hint
- known spatial assignment
- optional end-effector trajectory history
- candidate object names
- candidate grasp/skill affordances

Image prompting should be added after this offline candidate-selection baseline
is working.

## Dry Run

```bash
python3 scripts/casper_lite_evaluator.py --limit 20
```

The default backend is `rule`, a deterministic sanity-check backend. It is only
for testing the evaluator plumbing and should not be reported as a VLM result.

## Dump Prompts

```bash
python3 scripts/casper_lite_evaluator.py \
  --backend prompt_dump \
  --limit 20 \
  --output /tmp/casper_lite_prompts.jsonl
```

Each output row contains the full prompt, candidates, correct answer, and a
placeholder prediction. Use this mode to inspect or batch-submit prompts to a
VLM provider.

## External VLM Command

```bash
python3 scripts/casper_lite_evaluator.py \
  --backend command \
  --command "python3 scripts/casper_lite_vlm_predict_one.py --provider openai --model gpt-5" \
  --self-consistency-k 5 \
  --agreement-threshold 4
```

The command receives one prompt on stdin and must print JSON on stdout:

```json
{"intent_id": "soy_side", "confidence": 0.82, "reason": "The instruction asks to pour from soy milk."}
```

The evaluator parses the response, runs self-consistency voting, and writes:

- `results/casper_lite_predictions.jsonl`
- `results/casper_lite_metrics.json`

## Trajectory History

The evaluator includes trajectory context when an episode has either:

- `trajectory_history`: a list of end-effector samples
- `trajectory_summary`: a compact precomputed summary

You can keep trajectory data in a sidecar JSONL and merge it by `episode_id`:

```bash
python3 scripts/casper_lite_evaluator.py \
  --trajectory-jsonl /path/to/casper_trajectory_sidecar.jsonl \
  --backend prompt_dump \
  --limit 20
```

Accepted sidecar shape:

```json
{
  "episode_id": "milk_scene_01_top__semsp_001",
  "trajectory_history": [
    {"stamp": 0.0, "ee_position": {"x": 0.45, "y": 0.10, "z": 0.30}},
    {"stamp": 1.0, "ee_position": {"x": 0.55, "y": 0.08, "z": 0.25}}
  ]
}
```

The user-study dashboard probability log now records `trajectory_history` and
`trajectory_summary` for future runs. Older logs only contain aggregate path
statistics, so they cannot recover the original motion history.

## Replay Builder

Build CASPER-lite replay episodes from probability logs:

```bash
python3 scripts/build_casper_lite_replay.py \
  logs/user_study_probability_*.jsonl \
  --trial-logs logs/user_study_trials_*.jsonl \
  --output results/casper_lite_replay_episodes.jsonl
```

Then evaluate those replay episodes:

```bash
python3 scripts/casper_lite_evaluator.py \
  --replay-jsonl results/casper_lite_replay_episodes.jsonl \
  --backend prompt_dump \
  --limit 20
```

The builder matches probability rows to trial rows by session, participant,
condition, block, task, step, and timestamp when possible. If a row cannot be
matched to a target, it is skipped by default.

## Event-Triggered Visual Logging

For CASPER Section 3.2-style visual prompts, enable the lightweight observation
logger during a new run:

```bash
roslaunch tabletop_workspace_opt user_study.launch \
  launch_casper_observation_logger:=true \
  casper_save_image_width:=384 \
  casper_jpeg_quality:=70 \
  casper_min_save_interval_sec:=1.0 \
  casper_max_images_per_trial:=8
```

The logger writes:

- `logs/casper_observations_*.jsonl`
- `logs/casper_frames/*.jpg`

It saves downsized JPEGs only on decision-relevant events:

- trial start
- top-goal change
- top probability crossing the threshold
- selected grasp label change

This keeps storage bounded. With the defaults, each trial saves at most 8
images, each resized to 384 px wide and JPEG-compressed.

## DeepSeek Backend

DeepSeek can be used for the first text-only CASPER-lite baseline:

```bash
export DEEPSEEK_API_KEY="..."

python3 scripts/casper_lite_evaluator.py \
  --backend command \
  --command "python3 scripts/casper_lite_vlm_predict_one.py --provider deepseek --model deepseek-v4-pro" \
  --self-consistency-k 5 \
  --agreement-threshold 4
```

This uses DeepSeek's OpenAI-compatible Chat Completions endpoint and JSON Output
mode. It should be treated as a text-only commonsense intent selector unless a
vision-capable DeepSeek endpoint is added later.

For command-mode plumbing without a VLM API key:

```bash
python3 scripts/casper_lite_evaluator.py \
  --backend command \
  --command "python3 scripts/casper_lite_vlm_predict_one.py --provider heuristic" \
  --limit 20
```

The `heuristic` provider is only a local sanity check. It should not be reported
as a CASPER/VLM result.

## Next Additions

1. Add image prompt generation with candidate labels overlaid on the RGB frame.
2. Build a replay dataset directly from new probability logs with trajectory fields.
3. Wrap the evaluator backend in an online ROS node that publishes the same
   topics as `apriltag_intent_inference`.
