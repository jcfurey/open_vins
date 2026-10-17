#!/usr/bin/env python3
"""Reproducible OpenVINS benchmarks against known ground truth.

Sequences, their sources, estimator configs and ground truth are listed in
sequences.json next to this script. Typical use from the repository root:

  benchmark.py fetch --data ~/datasets --only tum_vi_room1
  benchmark.py run   --data ~/datasets --results /tmp/ov_bench --repeat 2
  benchmark.py check --results /tmp/ov_bench --baseline ov_eval/benchmark/baselines/x86_64.json

`run` replays every sequence serially with asl_serial_msckf (ASL/EuRoC folders) or
ros2_serial_msckf (ROS 2 bags), evaluates it with `error_singlerun --json`, and
records whether repeated runs produced bit-identical trajectories. `check` gates
accuracy against a stored baseline; --strict also requires identical trajectory
digests, which only holds for the same binaries on the same platform.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load_manifest(path):
    with open(path) as handle:
        return json.load(handle)


def select(manifest, only):
    if not only:
        return manifest
    missing = [name for name in only if name not in manifest]
    if missing:
        sys.exit(f"Unknown sequences: {', '.join(missing)}")
    return {name: manifest[name] for name in only}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_path(data, name, sequence):
    return Path(data).expanduser() / name / sequence["folder"]


def fetch(args):
    for name, sequence in select(load_manifest(args.manifest), args.only).items():
        target = sequence_path(args.data, name, sequence)
        if target.exists():
            print(f"[{name}] present at {target}")
            continue
        if not sequence.get("url"):
            print(f"[{name}] no public download; place the data at {target}")
            continue
        folder = target.parent
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / Path(sequence["url"]).name
        print(f"[{name}] downloading {sequence['url']}")
        subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(archive), sequence["url"]], check=True)
        if sequence.get("sha256"):
            actual = sha256(archive)
            if actual != sequence["sha256"]:
                sys.exit(f"[{name}] checksum mismatch: {actual} != {sequence['sha256']}")
            print(f"[{name}] sha256 verified")
        else:
            print(f"[{name}] sha256 {sha256(archive)} (not pinned in the manifest)")
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tar:
                tar.extractall(folder, filter="data")
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as package:
                package.extractall(folder)
        if not target.exists():
            sys.exit(f"[{name}] expected {target} after download")
        if archive.exists() and archive != target and not args.keep_archives:
            archive.unlink()
        print(f"[{name}] ready at {target}")


def find_executable(explicit, package, name):
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        candidate = Path(prefix) / "lib" / package / name
        if prefix and candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"Cannot find {name}; source the workspace or pass its path explicitly")


def environment():
    def git(*command):
        try:
            return subprocess.run(["git", "-C", str(REPO), *command], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as handle:
            cpu = next(line.split(":", 1)[1].strip() for line in handle if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "machine": platform.machine(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "system": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def run_logged(command, log, env=None):
    with open(log, "w") as handle:
        handle.write(" ".join(command) + "\n")
        handle.flush()
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode


def metrics(summary, evaluation):
    result = {
        "ate_pos_m": evaluation["ate"]["rmse_pos_m"],
        "ate_ori_deg": evaluation["ate"]["rmse_ori_deg"],
        "rpe": {str(item["segment_m"]): {"pos_m": item["median_pos_m"], "ori_deg": item["median_ori_deg"], "samples": item["samples"]}
                for item in evaluation["rpe"]},
        "matched_poses": evaluation["matched_poses"],
        "matched_duration_s": evaluation["matched_duration_s"],
        "nees_ori": (evaluation.get("nees") or {}).get("mean_ori"),
        "nees_pos": (evaluation.get("nees") or {}).get("mean_pos"),
        "frames_processed": summary["cameras"]["processed"],
        "frame_ms_mean": summary["timing"]["frame_ms_mean"],
        "frame_ms_p95": summary["timing"]["frame_ms_p95"],
        "peak_rss_mib": summary["memory"]["peak_rss_mib"],
        "digest": summary["trajectory"]["digest"],
    }
    return result


def simulation_config(sequence, out, verbosity):
    """Copy the simulation config folder and point it at the bundled trajectory, independent of the working directory."""
    source = (REPO / sequence["config"]).parent
    target = out / "config"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    config = target / Path(sequence["config"]).name
    text = config.read_text()
    text, count = re.subn(r'^sim_traj_path:.*$', f'sim_traj_path: "{REPO / sequence["trajectory"]}"', text, flags=re.MULTILINE)
    if count != 1:
        sys.exit(f"{sequence['config']} has no sim_traj_path to override")
    text = re.sub(r'^verbosity:.*$', f'verbosity: "{verbosity}"', text, flags=re.MULTILINE)
    config.write_text(text)
    return config


def run(args):
    manifest = select(load_manifest(args.manifest), args.only)
    evaluator = find_executable(args.evaluator, "ov_eval", "error_singlerun")
    results_dir = Path(args.results).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment(), "sequences": {}}
    env = dict(os.environ, MPLBACKEND="Agg")
    for name, sequence in manifest.items():
        data = sequence_path(args.data, name, sequence) if sequence["format"] != "sim" else REPO / sequence["trajectory"]
        if not data.exists():
            print(f"[{name}] skipped: no data at {data} (see `fetch`)")
            continue
        runs = []
        for index in range(args.repeat):
            out = results_dir / name / f"run{index}"
            out.mkdir(parents=True, exist_ok=True)
            config = str(REPO / sequence["config"])
            groundtruth = str(REPO / sequence["groundtruth"]) if sequence.get("groundtruth") else str(out / "groundtruth.txt")
            if sequence["format"] == "sim":
                runner = find_executable(args.sim_runner, "ov_msckf", "run_simulation")
                command = [runner, str(simulation_config(sequence, out, args.verbosity)), str(out)]
            elif sequence["format"] == "asl":
                runner = find_executable(args.runner, "ov_msckf", "asl_serial_msckf")
                command = [runner, config, str(data), str(out), "--verbosity", args.verbosity]
                if sequence.get("start"):
                    command += ["--start", str(sequence["start"])]
                if sequence.get("duration"):
                    command += ["--duration", str(sequence["duration"])]
            else:
                runner = find_executable(args.bag_runner, "ov_msckf", "ros2_serial_msckf")
                command = [runner, config, "--ros-args", "-p", f"path_bag:={data}", "-p", f"output_dir:={out}",
                           "-p", f"verbosity:={args.verbosity}", "-p", f"bag_start:={float(sequence.get('start') or 0)}",
                           "-p", f"bag_duration:={float(sequence.get('duration') or -1)}"]
            started = time.monotonic()
            code = run_logged(command, out / "log.txt", env)
            if code != 0:
                print(f"[{name}] run {index} failed with exit code {code}; see {out / 'log.txt'}")
                runs.append({"error": f"runner exit code {code}"})
                continue
            evaluate = [evaluator, sequence.get("align", "posyaw"), groundtruth, str(out / "trajectory.txt"), "--json", str(out / "eval.json")]
            if sequence.get("segments"):
                evaluate += ["--segments", ",".join(str(value) for value in sequence["segments"])]
            if run_logged(evaluate, out / "eval_log.txt", env) != 0:
                print(f"[{name}] evaluation {index} failed; see {out / 'eval_log.txt'}")
                runs.append({"error": "evaluation failed"})
                continue
            with open(out / "summary.json") as handle:
                summary = json.load(handle)
            with open(out / "eval.json") as handle:
                evaluation = json.load(handle)
            result = metrics(summary, evaluation)
            result["wall_s"] = time.monotonic() - started
            runs.append(result)
            print(f"[{name}] run {index}: ATE {result['ate_pos_m']:.4f} m / {result['ate_ori_deg']:.3f} deg, "
                  f"{result['frames_processed']} frames, {result['frame_ms_mean']:.2f} ms/frame, digest {result['digest']}")
        valid = [item for item in runs if "error" not in item]
        entry = {"runs": runs}
        if valid:
            entry["metrics"] = valid[0]
            entry["deterministic"] = len({item["digest"] for item in valid}) == 1 and len(valid) == len(runs)
            print(f"[{name}] deterministic across {len(valid)} run(s): {entry['deterministic']}")
        report["sequences"][name] = entry
    with open(results_dir / "results.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {results_dir / 'results.json'}")
    failed = [name for name, entry in report["sequences"].items() if "metrics" not in entry]
    return 1 if failed else 0


def within(value, reference, relative, absolute):
    return value <= reference * (1 + relative) + absolute


def check(args):
    with open(Path(args.results).expanduser() / "results.json") as handle:
        results = json.load(handle)
    baseline_path = Path(args.baseline)
    if args.update:
        stored = {"environment": results["environment"],
                  "sequences": {name: {"metrics": entry["metrics"]} for name, entry in results["sequences"].items() if "metrics" in entry}}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as handle:
            json.dump(stored, handle, indent=2)
            handle.write("\n")
        print(f"Updated {baseline_path}")
        return 0
    with open(baseline_path) as handle:
        baseline = json.load(handle)
    failures = []
    for name, reference in select(baseline["sequences"], args.only).items():
        entry = results["sequences"].get(name)
        if entry is None or "metrics" not in entry:
            print(f"[{name}] MISSING from results")
            failures.append(name)
            continue
        current, expected = entry["metrics"], reference["metrics"]
        problems = []
        if not within(current["ate_pos_m"], expected["ate_pos_m"], args.ate_tolerance, 0.005):
            problems.append(f"ATE position {current['ate_pos_m']:.4f} m vs {expected['ate_pos_m']:.4f} m")
        if not within(current["ate_ori_deg"], expected["ate_ori_deg"], args.ate_tolerance, 0.1):
            problems.append(f"ATE orientation {current['ate_ori_deg']:.3f} deg vs {expected['ate_ori_deg']:.3f} deg")
        for segment, rpe in expected["rpe"].items():
            now = current["rpe"].get(segment)
            if now is None or not within(now["pos_m"], rpe["pos_m"], args.rpe_tolerance, 0.005) or \
                    not within(now["ori_deg"], rpe["ori_deg"], args.rpe_tolerance, 0.1):
                problems.append(f"RPE {segment} m regressed")
        if current["matched_poses"] < 0.95 * expected["matched_poses"]:
            problems.append(f"coverage {current['matched_poses']} poses vs {expected['matched_poses']}")
        if not entry.get("deterministic", True):
            problems.append("repeated runs differ")
        if args.strict and current["digest"] != expected["digest"]:
            problems.append(f"digest {current['digest']} != {expected['digest']}")
        change = (current["ate_pos_m"] - expected["ate_pos_m"]) / expected["ate_pos_m"] * 100 if expected["ate_pos_m"] else 0.0
        status = "FAIL" if problems else "ok"
        print(f"[{name}] {status}: ATE {current['ate_pos_m']:.4f} m ({change:+.1f}%), {current['ate_ori_deg']:.3f} deg, "
              f"{current['frame_ms_mean']:.2f} ms/frame (baseline {expected['frame_ms_mean']:.2f})")
        for problem in problems:
            print(f"    - {problem}")
        if problems:
            failures.append(name)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(HERE / "sequences.json"))
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download and verify sequences")
    fetch_parser.add_argument("--data", required=True)
    fetch_parser.add_argument("--only", nargs="*")
    fetch_parser.add_argument("--keep-archives", action="store_true")

    run_parser = commands.add_parser("run", help="replay and evaluate sequences")
    run_parser.add_argument("--data", required=True)
    run_parser.add_argument("--results", required=True)
    run_parser.add_argument("--only", nargs="*")
    run_parser.add_argument("--repeat", type=int, default=2, help="runs per sequence; >1 also checks determinism")
    run_parser.add_argument("--verbosity", default="WARNING")
    run_parser.add_argument("--runner", help="asl_serial_msckf path (default: from the sourced workspace or PATH)")
    run_parser.add_argument("--bag-runner", help="ros2_serial_msckf path")
    run_parser.add_argument("--sim-runner", help="run_simulation path")
    run_parser.add_argument("--evaluator", help="error_singlerun path")

    check_parser = commands.add_parser("check", help="compare results with a baseline")
    check_parser.add_argument("--results", required=True)
    check_parser.add_argument("--baseline", required=True)
    check_parser.add_argument("--only", nargs="*", help="baseline sequences to check (default: all)")
    check_parser.add_argument("--ate-tolerance", type=float, default=0.10, help="allowed relative ATE increase")
    check_parser.add_argument("--rpe-tolerance", type=float, default=0.15, help="allowed relative RPE increase")
    check_parser.add_argument("--strict", action="store_true", help="also require identical trajectory digests")
    check_parser.add_argument("--update", action="store_true", help="write the results as the new baseline")

    args = parser.parse_args()
    return {"fetch": fetch, "run": run, "check": check}[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
