"""Builds the VLM intent-classification prompt (system + user messages).

The VLM sees a short teleoperation history and picks exactly one candidate
from the task state machine's ValidGoals (or UNCLEAR). It never outputs
motion.
"""
from typing import Any, Dict, List, Optional

from .vlm_client import image_content_part, text_content_part

SYSTEM_PROMPT = """You are the intent-recognition module of a shared-autonomy robot system. A
human is teleoperating a Sawyer arm to do a tabletop task. Your ONLY job is
to infer which single discrete subtask the human is currently trying to do,
chosen from a fixed multiple-choice list. You never output motion, poses, or
coordinates.

You are shown a short history of the teleoperation from a fixed third-person
camera facing the robot. Each history entry has a text line with the relative
timestamp, the end-effector world position, the gripper state, and the user's
commanded velocity direction. The most recent frame is annotated: green
markers lettered A, B, C, ... label the candidate objects, a green arrow
shows the end-effector's motion over the last ~1.5 seconds, and a translucent
cyan overlay highlights the robot's gripper.

Rules:
- Judge intent mainly from the direction and consistency of the recent
  motion (the arrow and the end-effector position trend) relative to the
  marked candidate objects.
- Be conservative. If the recent motion does not clearly commit to one
  option (barely moving, ambiguous between two markers, or retracting),
  answer "UNCLEAR".
- Options listed as recently rejected by the user are less likely; pick
  them only with strong evidence.
- Respond with STRICTLY ONE JSON object and nothing else. No markdown, no
  prose outside the JSON.

Response schema:
{"candidate_id": "<A|B|...|UNCLEAR>", "confidence": <0.0-1.0>, "reasoning": "<one short sentence>"}
"""

SKILL_VERBS = {
    "pick": "Pick up the {object}",
    "place": "Place the held object near the {reference}",
    "pour": "Pour the held object into the {reference}",
}

MARK_LETTERS = "ABCDEFGHJKLMNP"


def goal_description(goal_spec: Dict[str, Any]) -> str:
    """Human-readable description of a goal_spec (used in the MCQ and in
    rejection feedback)."""
    action = goal_spec.get("action", "?")
    if action == "pick":
        return SKILL_VERBS["pick"].format(object=goal_spec.get("object", "?"))
    dest = goal_spec.get("destination", {}) or {}
    ref = dest.get("reference", "target location")
    return SKILL_VERBS.get(action, action + " {reference}").format(
        reference=ref)


def candidate_text(letter: str, goal_spec: Dict[str, Any]) -> str:
    return "%s. %s (marker %s)" % (letter, goal_description(goal_spec),
                                   letter)


def build_candidates(goal_specs: List[Dict[str, Any]]):
    """[(letter, goal_spec)] with stable letters in list order."""
    return [(MARK_LETTERS[i], spec)
            for i, spec in enumerate(goal_specs[:len(MARK_LETTERS)])]


def build_messages(
    current_state: str,
    holding: Optional[str],
    frames,                      # List[HistoryRecord], oldest first
    annotated_image,             # BGR ndarray (latest frame, annotated)
    candidates,                  # [(letter, goal_spec)]
    rejected_descriptions: List[str],
    image_send_mode: str = "annotated_plus_text",
) -> List[Dict[str, Any]]:
    holding_desc = ("holding the %s" % holding) if holding \
        else "not holding anything"
    newest_stamp = frames[-1].stamp if frames else 0.0

    content: List[Dict[str, Any]] = [text_content_part(
        "Task state: %s. The robot is currently %s.\n\n"
        "Teleoperation history (oldest to newest):"
        % (current_state, holding_desc))]

    for record in frames[:-1]:
        line = record.state_line(record.stamp - newest_stamp)
        if image_send_mode == "all_images":
            content.append(text_content_part(line))
            content.append(image_content_part(record.image))
        else:
            content.append(text_content_part(line))
    if frames:
        content.append(text_content_part(
            frames[-1].state_line(0.0)
            + "  (frame below is annotated: markers + motion arrow)"))
        content.append(image_content_part(annotated_image))

    lines = ["", "Candidate subtasks (choose exactly one):"]
    for letter, spec in candidates:
        lines.append(candidate_text(letter, spec))
    lines.append("UNCLEAR. The intent is not yet clear from the motion.")
    lines.append("")
    lines.append("Recently rejected by the user (treat as unlikely): %s"
                 % (", ".join(rejected_descriptions) or "none"))
    lines.append("")
    lines.append("Answer with JSON only.")
    content.append(text_content_part("\n".join(lines)))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
