# VLM-Enhanced Intent Scoring for Fixed-Grasp Milk Selection

## Goal

We extended the fixed-grasp milk manipulation setup with a vision-language
scoring module. The task is to choose the correct milk carton and fixed grasp
type from a natural-language instruction under a controlled tabletop scene.

The three cartons are:
- `whole_milk`
- `oat_milk`
- `soy_milk`

The predefined fixed grasps are:
- `*_top` for pickup
- `*_side` for pour

The core question was:

Can a frozen VLM backbone improve candidate scoring beyond the original
intent-only pipeline, especially when the instruction requires visual grounding?

This project is intentionally not a full end-to-end vision-language-action
(VLA) policy. Low-level execution remains fixed. The learned component only
improves the high-level decision stage:

`Image + Instruction -> Candidate / Object Score -> Best Fixed Grasp -> Execution`

So the most accurate description is:
- VLM-enhanced intent scoring
- multimodal candidate classification
- visual grounding for fixed-grasp selection

## Data Collection Setup

### Scene Design

We intentionally kept the robot-side fixed grasp definitions unchanged, but
removed the AprilTag-based pose anchoring from the learning setup. Instead, we
fixed the tabletop workspace and recorded RGB images under multiple layouts.

Each recorded scene contains:
- 3 milk cartons placed in left / center / right slots
- 3 camera views: `top`, `side`, `lean`
- scene metadata:
  - `scene_id`
  - `view_id`
  - `slot_assignment`
  - `image_path`

### Recording Pipeline

Custom recording support was added in:
- [record_candidate_dataset.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/record_candidate_dataset.py)

This script saves:
- RGB image snapshots
- instruction labels
- slot assignments
- candidate metadata

We ultimately used PNG + JSONL recording instead of rosbag because the bag file
size was too large for the available disk budget.

### Dataset Status

Current recorded data includes:
- 15 distinct scene/view images
- multiple base layouts with different left / center / right assignments
- synthetic prompt expansion for larger training sets

Generated dataset files:
- [episodes_semantic_spatial.jsonl](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/data/milk_candidate_cls/episodes_semantic_spatial.jsonl)
- [candidate_samples_semantic_spatial.jsonl](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/data/milk_candidate_cls/candidate_samples_semantic_spatial.jsonl)

Prompt generation is handled by:
- [generate_semantic_spatial_dataset.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/scripts/generate_semantic_spatial_dataset.py)

## Why This Is Not a Full VLA

CLIP and SigLIP are useful vision-language backbones, but in this project they
are used as frozen multimodal feature extractors, not as action-generating
robot policies.

The current system does not:
- predict continuous robot actions
- learn a manipulation trajectory end-to-end
- replace the fixed grasp execution controller

Instead, it:
- scores discrete candidates
- classifies the target object from image + language
- selects the best predefined fixed grasp

This distinction matters because a true VLA such as TinyVLA or another
small-scale VLA is designed for end-to-end policy learning, while our problem
here is better formulated as multimodal scoring and classification.

## Experiment Progression

### Attempt 1: Candidate-Level Binary Classification

We first expanded each episode into 6 candidate rows and trained a binary
classifier over candidate correctness.

Representative result:
- accuracy around `0.83`
- precision / recall / F1 collapsed toward `0`

Diagnosis:
- the task was heavily imbalanced
- 5 out of 6 candidates are negatives
- the model could predict negatives everywhere and still get superficially high
  accuracy

Conclusion:
- binary candidate classification was not a meaningful formulation

Notebook:
- [vlm_candidate_classifier.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_candidate_classifier.ipynb)

### Attempt 2: Episode-Level 6-Way Candidate Ranking

We then switched to episode-level ranking, where each instruction-image pair
must select exactly one of 6 candidates.

Compared models:
- text-only CLIP baseline
- image+text CLIP
- image+text SigLIP

Observation:
- text-only baseline became surprisingly strong
- SigLIP performed best on the early mixed dataset

Diagnosis:
- object identity and task semantics were leaking into the language
- prompts such as `Pick the whole milk.` or `Pick the dairy milk.` let the
  model solve the task from text matching

Conclusion:
- the benchmark still had strong language shortcuts

Notebook:
- [vlm_candidate_classifier_enhanced.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_candidate_classifier_enhanced.ipynb)

### Attempt 3: Spatial-Only 6-Way Ranking

To reduce language shortcuts, we created spatial prompts such as:
- `Pick the carton on the left.`
- `Pour from the carton in the center.`

We also weakened candidate text to only contain grasp information:
- `Candidate grasp type: top grasp.`
- `Candidate grasp type: side grasp.`

Result:
- all models converged to about `0.333`

Diagnosis:
- this was not random failure
- the 6-way task still contains a built-in shortcut:
  - `pickup -> *_top`
  - `pour -> *_side`
- therefore the effective baseline becomes `1/3`, not `1/6`

Conclusion:
- the 6-way spatial benchmark was still structurally flawed

Notebook:
- [vlm_spatial_only_comparison.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_spatial_only_comparison.ipynb)

### Attempt 4: Spatial Object Selection (Final Formulation)

We finally reformulated the task as **3-way object classification**:
- `whole`
- `oat`
- `soy`

The instruction remains unchanged, for example:
- `Pick the carton on the left.`
- `Pour from the carton in the center.`

The predicted object is then mapped back to the final grasp candidate:
- `pickup -> <object>_top`
- `pour -> <object>_side`

This removes the previous `pick/pour -> top/side` shortcut from the prediction
space.

Notebook:
- [vlm_spatial_object_comparison.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_spatial_object_comparison.ipynb)

## Prompt Engineering

The first spatial prompt set was too templated. We expanded it into a more
natural paraphrase set while preserving unique ground-truth labels.

Examples:
- `Pick the carton on the left.`
- `Pick up the left carton.`
- `Grab the leftmost carton.`
- `Choose the carton on the left side.`
- `Take the carton on the far left.`
- `Lift the left-side carton.`

For pouring:
- `Pour from the carton on the right.`
- `Use the right carton for pouring.`
- `Grasp the rightmost carton for pouring.`
- `Pour using the carton on the right side.`

This increased prompt diversity and reduced the chance that the model simply
memorizes a small set of rigid templates.

After prompt expansion, the generated prompt counts became:
- `spatial_prompt = 540`
- `semantic_prompt = 180`
- `semantic_spatial_prompt = 90`

## Final Comparison

Final comparison was run on the **spatial object selection** benchmark with the
new paraphrased spatial prompts.

| Model | Best Val Object Top-1 | Best Val Candidate Top-1 |
|------|------------------------|--------------------------|
| `image_text_clip` | `0.9028` | `0.9028` |
| `image_text_siglip` | `0.8403` | `0.8403` |
| `text_only_clip` | `0.8333` | `0.8333` |
| `text_only_siglip` | `0.8333` | `0.8333` |

The final deployed runtime model for the shared-autonomy upgrade is the
`image_text_clip` 3-way object classifier. We kept SigLIP as a comparison
baseline, but CLIP was chosen for integration because it gave the strongest
final result and was simpler to deploy inside the ROS pipeline.

## Shared-Autonomy Fusion Integration

After training, the classifier was integrated into the existing
`fixed_grasp_intent_with_pour` pipeline as a semantic correction term rather
than a new end-to-end controller.

### Original Pipeline

Before this upgrade, the fixed-grasp shared-autonomy flow was:

`Teleop motion history -> candidate probability -> best fixed grasp -> execution`

This works when motion intent is clean, but it can fail when the operator
briefly moves toward the wrong object. In that case, proximity and trajectory
history can dominate the score even if the language instruction points to a
different target.

### Upgraded Pipeline

The upgraded flow is:

`RGB image + instruction -> object semantic score`

`teleop candidate score + semantic object score -> fused object score -> fixed grasp execution`

The key design decision was to keep the low-level execution unchanged:
- `pickup -> <object>_top`
- `pour -> <object>_side`

So the new learned component only changes the selection stage, not the motion
controller.

### Runtime Modules Added

Semantic scoring and fusion support were added in:
- [shared_autonomy_pregrasp_selector.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/shared_autonomy_pregrasp_selector.py)
- [intent_score_fusion.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/intent_score_fusion.py)
- [vlm_semantic_scorer.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/vlm_semantic_scorer.py)
- [milk_object_classifier_runtime.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/milk_object_classifier_runtime.py)
- [evaluate_intent_fusion_log.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/evaluate_intent_fusion_log.py)

Launch files updated for the demo path:
- [fixed_grasp_intent.launch](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/launch/fixed_grasp_intent.launch)
- [fixed_grasp_intent_with_pour.launch](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/launch/fixed_grasp_intent_with_pour.launch)

### Fusion Formula

For the project demo we use weighted-sum fusion:

`P_final(i) = alpha * P_teleop(i) + beta * P_vlm(i)`

where `i` is an object label in:
- `whole_milk`
- `oat_milk`
- `soy_milk`

We chose weighted-sum fusion instead of multiplicative fusion because:
- it is easier to debug
- it is numerically stable under small-data uncertainty
- it lets us tune teleop / VLM influence directly for demo purposes

Typical demo settings:
- `alpha = 0.7`, `beta = 0.3` for conservative correction
- `alpha = 0.5`, `beta = 0.5` for a stronger visible correction effect

### What the Runtime Model Outputs

The deployed classifier does not output robot actions. It outputs normalized
object probabilities:

`{whole_milk, oat_milk, soy_milk}`

These object probabilities are then mapped back to fixed grasp candidates
according to the task:
- for `pickup`, use `*_top`
- for `pour`, use `*_side`

This point is important for presentation: the label is a semantic target
object, not a continuous grasp pose regression target.

## Demo Configuration

### Recommended Scene Layout

For the main demo, the clearest layout is:
- left: `soy_milk`
- center: `oat_milk`
- right: `whole_milk`

Recommended instruction:
- `Pour from the carton on the right.`

Correct semantic target:
- `whole_milk`

Correct final fixed grasp:
- `whole_side`

### Intended Failure Mode

The baseline demo should intentionally create a motion-bias error:
- the instruction refers to the right carton
- the operator first moves toward the left carton
- teleop-only intent rises for the wrong object

This creates the before/after story we want:
- `teleop-only`: wrong object probability rises due to motion bias
- `teleop + VLM`: the semantic score suppresses the wrong object and helps
  recover the correct target

This is a better demo than a fully clean trajectory because if teleop already
selects the correct object throughout the trial, the fusion term has no visible
correction work to do.

## Logging and Evaluation

To support debugging and presentation, the upgraded selector logs:
- `teleop_object_probs`
- `vlm_object_probs`
- `fused_object_probs`
- selected object
- selected grasp
- expected object

Example output fields in the CSV log:
- `teleop_top_object`
- `selected_object`
- `vlm_top_object`
- `teleop_object_probs_json`
- `vlm_object_probs_json`
- `fused_object_probs_json`

This makes it possible to show:
- when the teleop-only baseline drifts toward the wrong object
- whether semantic fusion changed the final selected object
- how much confidence moved between objects over time

## Runtime Procedure

### 1. Train / Export the Runtime Classifier

The deployment checkpoint is produced by:
- [train_milk_object_classifier.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/scripts/train_milk_object_classifier.py)

Example:

```bash
/usr/bin/python3 src/tabletop_workspace_opt/scripts/train_milk_object_classifier.py \
  --output /home/gyanig/catkin_ws/src/tabletop_workspace_opt/outputs/vlm_object_classifier_runtime/image_text_clip_object_classifier.pt
```

### 2. Set the Runtime Checkpoint

```bash
export MILK_OBJECT_CLASSIFIER_CKPT=/home/gyanig/catkin_ws/src/tabletop_workspace_opt/outputs/vlm_object_classifier_runtime/image_text_clip_object_classifier.pt
```

### 3. Run the Teleop-Only Baseline

```bash
roslaunch tabletop_workspace_opt fixed_grasp_intent_with_pour.launch \
  fusion_enabled:=false \
  task_action_filter:=pour \
  semantic_instruction:="Pour from the carton on the right." \
  expected_object_name:=whole_milk \
  intent_log_csv:=/tmp/teleop_only_right.csv
```

### 4. Run the Teleop + VLM Fusion Version

```bash
roslaunch tabletop_workspace_opt fixed_grasp_intent_with_pour.launch \
  fusion_enabled:=true \
  task_action_filter:=pour \
  semantic_backend:=callable \
  semantic_callable:=milk_object_classifier_runtime:predict_object_scores \
  semantic_image_topic:=/camera/color/image_raw \
  fusion_method:=weighted_sum \
  fusion_alpha:=0.5 \
  fusion_beta:=0.5 \
  semantic_instruction:="Pour from the carton on the right." \
  expected_object_name:=whole_milk \
  intent_log_csv:=/tmp/teleop_vlm_right.csv
```

### 5. Compare the Logs

```bash
/usr/bin/python3 /home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/evaluate_intent_fusion_log.py \
  --baseline-csv /tmp/teleop_only_right.csv \
  --fusion-csv /tmp/teleop_vlm_right.csv \
  --expected-object whole_milk \
  --output-csv /tmp/fusion_summary_right.csv
```

## Presentation Positioning

The most accurate way to describe the final system is:

**A lightweight vision-language semantic scoring module fused into a fixed-grasp shared-autonomy selector.**

This is more defensible than calling it a full generative VLM or a VLA policy,
because:
- the model does not generate robot trajectories
- the model does not regress grasp pose directly from pixels
- the model outputs semantic object scores, not low-level control
- execution remains based on predefined fixed grasps

So if asked about labels, the correct answer is:
- the label is the correct semantic target object or fixed grasp candidate
- not a grasp pose annotation drawn directly on the image

## Interpretation

### What Worked

1. Replacing 6-way candidate ranking with 3-way object classification made the
   benchmark much more meaningful.

2. Spatial-only instructions reduced direct object-identity leakage from the
   prompt.

3. Prompt paraphrasing improved the benchmark quality by reducing rigid template
   memorization.

4. Image-conditioned models finally outperformed text-only baselines:
   - `image_text_clip > text_only_clip`
   - `image_text_siglip > text_only_siglip`

This is the main evidence that visual information is contributing to the final
decision at the intent scoring stage.

### What Did Not Work

1. Candidate-level binary classification was dominated by class imbalance.

2. Mixed semantic prompts caused strong text shortcuts.

3. 6-way spatial ranking still retained a hidden `1/3` grasp-type shortcut.

## Key Takeaway

The main result is not simply that a VLM can score grasp candidates.

The stronger result is:

After removing language shortcuts, reformulating the task as object selection,
and increasing spatial prompt diversity, image-conditioned VLM models achieved
better performance than text-only baselines on fixed-grasp milk-carton
selection.

This supports the claim that the final system is using visual grounding, not
just language matching, at the candidate selection stage.

## Positioning

The most accurate project description is:

**A VLM-enhanced intent scoring module for fixed-grasp shared autonomy.**

This is stronger and more defensible than calling the method a full VLA policy,
because:
- execution is still handled by predefined fixed grasps
- the learned component is a classifier / scorer
- the model improves decision quality before grasp execution

## Future Work

A natural next step would be to replace the current scoring module with a
lightweight VLA policy such as TinyVLA or another small VLA architecture.

That future version would require:
- trajectory-level robot action data
- robot state / proprioception
- action supervision rather than only scene-level labels

That is outside the current experiment scope, which focuses on improving
candidate selection within an existing fixed-grasp pipeline.

## Relevant Files

Data / generation:
- [record_candidate_dataset.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/src/shared_autonomy/record_candidate_dataset.py)
- [expand_candidate_dataset.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/scripts/expand_candidate_dataset.py)
- [generate_spatial_instruction_dataset.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/scripts/generate_spatial_instruction_dataset.py)
- [generate_semantic_spatial_dataset.py](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/scripts/generate_semantic_spatial_dataset.py)

Notebooks:
- [vlm_candidate_classifier.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_candidate_classifier.ipynb)
- [vlm_candidate_classifier_enhanced.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_candidate_classifier_enhanced.ipynb)
- [vlm_spatial_only_comparison.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_spatial_only_comparison.ipynb)
- [vlm_spatial_object_comparison.ipynb](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/notebooks/vlm_spatial_object_comparison.ipynb)
