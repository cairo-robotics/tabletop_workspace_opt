"""Experiment provenance helpers shared by evaluation scripts."""
from __future__ import annotations

import datetime
import hashlib
import os
import subprocess


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(project_root):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root,
            text=True, stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_se3_sa_metadata(project_root, args, selected_tiers, *,
                          script_rel="scripts/eval/compare_se3_sa_3d.py"):
    commit, dirty = git_state(project_root)
    grasp_lib_path = os.path.join(
        project_root, "config", "grasp_poses_3d.yaml")
    input_paths = [
        grasp_lib_path,
        os.path.join(project_root, script_rel),
        os.path.join(project_root, "scripts", "eval", "run_sa_headless.py"),
        os.path.join(project_root, "src", "envopt", "grasp_feasibility.py"),
        os.path.join(project_root, "src", "envopt", "se3_observers.py"),
        os.path.join(project_root, "src", "envopt", "layout_sampling.py"),
        os.path.join(project_root, "src", "experiments", "se3_catalog.py"),
    ]
    for task_rel, scene, _, _ in selected_tiers:
        input_paths.extend([
            os.path.join(project_root, task_rel),
            os.path.join(project_root, "config", "scenes", f"{scene}.yaml"),
        ])
        p = os.path.join(project_root, "config", "scenes",
                         f"{scene}_se3_me_optimized.yaml")
        if os.path.exists(p):
            input_paths.append(p)
    return {
        "run_id": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"),
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "git_commit": commit,
        "dirty_worktree": dirty,
        "script": script_rel,
        "parameters": {
            "beta": args.beta, "threshold": args.threshold,
            "noise": args.noise, "control_rate": args.control_rate,
            "lambda_R": args.lambda_R, "arrival_dist": args.arrival_dist,
            "goal_timeout": 30.0, "max_speed": 0.05,
            "seed": args.seed, "n_random": args.n_random,
            "random_max_orderings": args.random_max_orderings,
        },
        "input_hashes": {
            os.path.relpath(p, project_root): sha256_file(p)
            for p in sorted(set(input_paths))
        },
    }


def metadata_compatible(old, new):
    """Compare fields that must match when independently produced arms merge."""
    old_exp, new_exp = old.get("experiment", {}), new.get("experiment", {})
    common_keys = ("beta", "threshold", "noise", "control_rate",
                   "lambda_R", "arrival_dist", "goal_timeout", "max_speed",
                   "seed")
    old_params = old_exp.get("parameters", {})
    new_params = new_exp.get("parameters", {})
    return (old.get("schema_version") == new.get("schema_version") == 2
            and all(old_params.get(k) == new_params.get(k)
                    for k in common_keys)
            and old_exp.get("input_hashes") == new_exp.get("input_hashes")
            and old_exp.get("git_commit") == new_exp.get("git_commit"))
