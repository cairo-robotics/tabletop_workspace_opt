"""CASPER-style VLM intent inference for shared autonomy.

Self-contained (no vlm_robot_controller dependency): the VLM observes
teleoperation history and classifies which ValidGoal the user intends;
it never produces motion.
"""
