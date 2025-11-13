#!/usr/bin/env python3
import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

PROFILES = {
    "num_workers": [1, 2],
    "rollout_length": [32, 64],
    "min_timesteps_per_update": [1024, 2048],
}


def run_once(config_path: Path, overrides: dict) -> dict:
    sweep_cfg = yaml.safe_load(config_path.read_text())
    sweep_cfg.setdefault("parallel", {}).update(
        {
            "num_workers": overrides["num_workers"],
            "rollout_length": overrides["rollout_length"],
        }
    )
    sweep_cfg.setdefault("training", {}).update(
        {"min_timesteps_per_update": overrides["min_timesteps_per_update"]}
    )
    temp_cfg = Path(tempfile.mkstemp(suffix=".yaml")[1])
    temp_cfg.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False))

    cmd = [
        "python",
        "experiments/train_mogcrl_trpo_qp.py",
        "--config",
        str(temp_cfg),
        "--iters",
        "2",
    ]
    print(f"[Sweep] running {cmd}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    temp_cfg.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"command failed: {cmd}")
    return {"stdout": proc.stdout, "stderr": proc.stderr}


def main():
    parser = argparse.ArgumentParser(description="吞吐 sweep")
    parser.add_argument("--config", type=str, default="configs/mogcrl.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    results = []
    for combo in itertools.product(
        PROFILES["num_workers"],
        PROFILES["rollout_length"],
        PROFILES["min_timesteps_per_update"],
    ):
        overrides = {
            "num_workers": combo[0],
            "rollout_length": combo[1],
            "min_timesteps_per_update": combo[2],
        }
        output = run_once(config_path, overrides)
        results.append({"overrides": overrides, "logs": output})

    out_path = Path("results") / "sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Sweep 结果已写入 {out_path}")


if __name__ == "__main__":
    main()

