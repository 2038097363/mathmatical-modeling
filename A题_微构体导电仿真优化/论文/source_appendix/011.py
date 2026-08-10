from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path


def _discover_project_root(script_path: Path) -> Path:
    # 优先使用环境变量，否则向上查找项目标志目录，避免写死机器绝对路径。
    configured = os.environ.get("MCM_PROJECT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else script_path.resolve().parents
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "公共代码").is_dir() and (root / "问题").is_dir():
            return root
    raise RuntimeError("无法定位项目根目录；请设置 MCM_PROJECT_ROOT")


ROOT = _discover_project_root(Path(__file__))
DEFAULT_REGISTRY = ROOT / "结果注册表.json"
DEFAULT_TEX = ROOT / "论文" / "generated" / "results.tex"
MACRO_PATTERN = re.compile(r"^[A-Za-z]+$")


def read_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    if not path.exists():
        return {"project": ROOT.name, "results": {}}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("results"), dict):
        raise ValueError(f"结果注册表结构错误: {path}")
    return value


def write_registry(value: dict, path: Path = DEFAULT_REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# 关键：登记数值、来源、校验状态和可选 LaTeX 宏。
def register_result(
    key: str,
    *,
    question: int,
    value,
    unit: str = "",
    source_script: str,
    source_artifact: str,
    validation: str,
    latex_macro: str | None = None,
    formatted: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    if latex_macro and not MACRO_PATTERN.fullmatch(latex_macro):
        raise ValueError("LaTeX 宏名只能包含英文字母")
    registry = read_registry(registry_path)
    registry["results"][key] = {
        "question": question,
        "value": value,
        "formatted": formatted,
        "unit": unit,
        "source_script": source_script,
        "source_artifact": source_artifact,
        "validation": validation,
        "latex_macro": latex_macro,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_registry(registry, registry_path)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(char, char) for char in value)


# 关键：将已登记结果导出为论文可直接引用的 LaTeX 宏。
def export_latex(
    registry_path: Path = DEFAULT_REGISTRY,
    output_path: Path = DEFAULT_TEX,
) -> int:
    registry = read_registry(registry_path)
    lines = ["% 由公共代码/result_registry.py自动生成，请勿手工修改。"]
    count = 0
    for key in sorted(registry["results"]):
        item = registry["results"][key]
        macro = item.get("latex_macro")
        if not macro:
            continue
        shown = item.get("formatted")
        if shown is None:
            shown = str(item.get("value", ""))
        shown = latex_escape(shown + item.get("unit", ""))
        lines.append(f"\\providecommand{{\\{macro}}}{{{shown}}}")
        count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


if __name__ == "__main__":
    print(f"已导出 {export_latex()} 个 LaTeX 结果宏")
