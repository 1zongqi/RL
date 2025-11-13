#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="切换 MOGCRL 配置中的 active_profile")
    parser.add_argument("--config", type=str, default="configs/mogcrl.yaml", help="配置文件路径")
    parser.add_argument("--name", type=str, required=True, help="profile 名称")
    args = parser.parse_args()

    path = Path(args.config)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    data = yaml.safe_load(path.read_text())
    profiles = (data.get("profiles") or {})
    if args.name not in profiles:
        raise ValueError(f"未找到 profile '{args.name}'，可用: {list(profiles.keys())}")

    data["active_profile"] = args.name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"active_profile 已切换为 '{args.name}'（文件: {path})")


if __name__ == "__main__":
    main()

