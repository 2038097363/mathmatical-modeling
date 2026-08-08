from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    with (ROOT / "项目配置.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_args(question_count: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按依赖顺序复现各问结果")
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--to-step", type=int, default=question_count)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = load_config()
    questions = sorted(config["questions"], key=lambda item: int(item["id"]))
    args = parse_args(len(questions))
    if args.from_step < 1 or args.to_step > len(questions):
        raise ValueError("运行范围超出问题编号")
    if args.from_step > args.to_step:
        raise ValueError("--from-step 不能大于 --to-step")

    environment = os.environ.copy()
    environment["MATH_MODELING_PROJECT_ROOT"] = str(ROOT)
    environment["MATH_MODELING_SEED"] = str(config["random_seed"])

    for question in questions:
        index = int(question["id"])
        if not args.from_step <= index <= args.to_step:
            continue
        script = ROOT / question["script"]
        if not script.is_file():
            raise FileNotFoundError(script)
        runs = question.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"问题{index}缺少正式运行参数")
        for run_number, run in enumerate(runs, start=1):
            command_args = run.get("args")
            if not isinstance(command_args, list) or not all(
                isinstance(item, str) for item in command_args
            ):
                raise TypeError(f"问题{index}第{run_number}组参数必须是字符串列表")
            run_name = str(run.get("name", f"运行{run_number}"))
            command = [sys.executable, str(script), *command_args]
            print(
                f"[问题{index}/{run_name}] {subprocess.list2cmdline(command)}",
                flush=True,
            )
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=environment, check=True)

    if args.run_tests:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        print("[测试]", subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
