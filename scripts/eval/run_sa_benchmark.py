#!/usr/bin/env python3
"""Benchmark shared autonomy across optimized and unoptimized workspaces.

Runs multiple trials with different task orders and collects metrics:
- Time to infer each goal
- Inference confidence at trigger
- Number of retries needed
- Total task completion time

Usage:
    python3 scripts/eval/run_sa_benchmark.py [--trials 3]
"""
import sys
import os
import glob
import subprocess
import time
import re
import json
import argparse

def run_sa_trial(task_config, scene, noise=0.005, threshold=0.7, beta=5.0,
                 max_speed=0.15, control_rate=20, timeout=300):
    """Run one shared autonomy trial and parse the output.

    Returns dict with metrics or None on failure.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runner = os.path.join(pkg_root, "src", "shared_autonomy",
                          "shared_autonomy_runner.py")
    config = os.path.join(pkg_root, task_config)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DISPLAY"] = ":1"

    cmd = [
        sys.executable, runner, config,
        "--noise", str(noise),
        "--threshold", str(threshold),
        "--beta", str(beta),
        "--max-speed", str(max_speed),
        "--control-rate", str(control_rate),
        "--scene", scene,
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env,
                                cwd=pkg_root)
        elapsed = time.time() - start
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed": timeout, "steps": [],
                "total_steps": 0, "retries": 0}

    # Parse output
    metrics = {
        "status": "completed" if result.returncode == 0 else "failed",
        "elapsed": round(elapsed, 1),
        "steps": [],
        "total_steps": 0,
        "retries": 0,
    }

    # Extract step completions with inference time
    for m in re.finditer(
            r"INTENT INFERRED: (\S+) \(p=([\d.]+), t=([\d.]+)s\)", output):
        metrics["steps"].append({
            "goal": m.group(1),
            "confidence": float(m.group(2)),
            "inference_time": float(m.group(3)),
        })
    # Fallback for old format without time
    if not metrics["steps"]:
        for m in re.finditer(
                r"INTENT INFERRED: (\S+) \(p=([\d.]+)\)", output):
            metrics["steps"].append({
                "goal": m.group(1),
                "confidence": float(m.group(2)),
                "inference_time": 0.0,
            })

    # Count timeouts (retries)
    metrics["retries"] = output.count("Timeout reaching goal")

    # Extract total steps
    m = re.search(r"Steps completed: (\d+)", output)
    if m:
        metrics["total_steps"] = int(m.group(1))

    # Extract final state
    m = re.search(r"Final state: (\S+)", output)
    if m:
        metrics["final_state"] = m.group(1)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per configuration")
    args = parser.parse_args()

    # Benchmark configurations: each entry is (scene_label, scene_name, task_list)
    # scene_label: human-readable label for grouping results
    # scene_name: MuJoCo scene name passed to roslaunch and --scene flag
    # task_list: list of (task_name, task_config_path) to run on this scene
    benchmark_configs = [
        # Breakfast scenes
        ("breakfast_unoptimized", "scene_breakfast", [
            ("full_breakfast_sa", "config/tasks/full_breakfast_sa.yaml"),
            ("set_table_sa", "config/tasks/set_table_sa.yaml"),
        ]),
        ("breakfast_optimized", "scene_breakfast_optimized", [
            ("full_breakfast_sa", "config/tasks/full_breakfast_sa.yaml"),
            ("set_table_sa", "config/tasks/set_table_sa.yaml"),
        ]),
        # Desk scenes
        ("desk_unoptimized", "scene_desk", [
            ("desk_organize_v2_sa", "config/tasks/desk_organize_v2_sa.yaml"),
        ]),
        ("desk_optimized", "scene_desk_optimized", [
            ("desk_organize_v2_sa", "config/tasks/desk_organize_v2_sa.yaml"),
        ]),
    ]

    results = []

    # Group runs by scene so we only restart sim when scene changes
    current_scene = None

    for scene_label, scene_name, task_list in benchmark_configs:
        if scene_name != current_scene:
            print(f"\n*** Launching sim for scene: {scene_name} ***")
            # Kill any existing sim
            os.system("pkill -f simulation_server.py 2>/dev/null; "
                       "pkill -f move_to_cartesian 2>/dev/null; "
                       "pkill -f roslaunch 2>/dev/null; "
                       "pkill -f rosmaster 2>/dev/null; "
                       "pkill -f rviz 2>/dev/null; sleep 3")
            # Launch sim with this scene
            launch_cmd = (
                f"nohup bash -c 'export DISPLAY=:1 && "
                f"source /home/yi-shiuan/sawyer_ws/devel_isolated/"
                f"tabletop_workspace_opt/setup.bash && "
                f"export ROS_PACKAGE_PATH=\"/home/yi-shiuan/sawyer_ws/src:"
                f"$ROS_PACKAGE_PATH\" && "
                f"roslaunch tabletop_workspace_opt sim_moveit.launch "
                f"scene_name:={scene_name}' "
                f"&>/tmp/sim_bench_{scene_name}.log &"
            )
            os.system(launch_cmd)
            print("  Waiting 40s for sim to start...")
            time.sleep(40)
            current_scene = scene_name

        for task_name, task_config in task_list:
            for trial in range(1, args.trials + 1):
                label = f"{task_name}/{scene_label}/trial{trial}"
                print(f"\n{'=' * 60}")
                print(f"  Running: {label}")
                print(f"{'=' * 60}")

                metrics = run_sa_trial(
                    task_config, scene_name,
                    noise=0.005, threshold=0.7, beta=5.0,
                    max_speed=0.15, control_rate=20,
                    timeout=180)

                metrics["task"] = task_name
                metrics["scene"] = scene_label
                metrics["trial"] = trial
                results.append(metrics)

                print(f"  Status: {metrics['status']}")
                print(f"  Elapsed: {metrics['elapsed']}s")
                print(f"  Steps: {metrics['total_steps']}")
                print(f"  Retries: {metrics['retries']}")
                if metrics.get("steps"):
                    avg_conf = sum(s["confidence"] for s in metrics["steps"]) / len(metrics["steps"])
                    inf_times = [s.get("inference_time", 0) for s in metrics["steps"]]
                    avg_inf = sum(inf_times) / len(inf_times) if inf_times else 0
                    print(f"  Avg confidence: {avg_conf:.3f}")
                    print(f"  Avg inference time per goal: {avg_inf:.2f}s")
                    for s in metrics["steps"]:
                        print(f"    {s['goal']}: p={s['confidence']:.3f}, t={s.get('inference_time', 0):.1f}s")

    # Clean up sim
    os.system("pkill -f simulation_server.py 2>/dev/null; "
               "pkill -f move_to_cartesian 2>/dev/null; "
               "pkill -f roslaunch 2>/dev/null; "
               "pkill -f rosmaster 2>/dev/null; "
               "pkill -f rviz 2>/dev/null")

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")

    # Group by task + scene
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = f"{r['task']}/{r['scene']}"
        grouped[key].append(r)

    print(f"\n{'Config':<40} {'Steps':>6} {'Retries':>8} {'Total(s)':>8} {'InfTime':>8} {'AvgConf':>8} {'Pass':>5}")
    print("-" * 84)

    summary_data = {}
    for key in sorted(grouped.keys()):
        runs = grouped[key]
        completed = [r for r in runs if r["status"] == "completed"]
        all_steps = sum(r["total_steps"] for r in runs) / len(runs)
        all_retries = sum(r["retries"] for r in runs) / len(runs)
        all_times = sum(r["elapsed"] for r in runs) / len(runs)
        all_confs = []
        all_inf_times = []
        for r in runs:
            for s in r.get("steps", []):
                all_confs.append(s["confidence"])
                all_inf_times.append(s.get("inference_time", 0.0))
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
        avg_inf_time = sum(all_inf_times) / len(all_inf_times) if all_inf_times else 0
        pass_rate = len(completed) / len(runs)

        inf_str = f"{avg_inf_time:.2f}s"
        print(f"  {key:<38} {all_steps:>6.1f} {all_retries:>8.1f} "
              f"{all_times:>8.1f} {inf_str:>8} {avg_conf:>8.3f} {pass_rate:>5.0%}")

        summary_data[key] = {
            "avg_steps": all_steps,
            "avg_retries": all_retries,
            "avg_total_time": all_times,
            "avg_inference_time_per_goal": avg_inf_time,
            "avg_confidence": avg_conf,
            "pass_rate": pass_rate,
            "n_trials": len(runs),
        }

    # Save results
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(pkg_root, "results", "sa_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trials": results, "summary": summary_data}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
