# A题微构体导电仿真优化项目

所有命令均从本目录运行。LaTeX/XeLaTeX 是论文权威构建链。包含 24 个真实程序的完整稿为
`论文/A题论文.pdf`。问题四 50000 次正式确认与独立完整域审计均已通过；
其冻结参数、源码、500 个分片、合并结果、日志和审计文件是正式证据，不得移动或修改。

## 权威目录

- `00_赛题与附件/`：题面、附件、论文模板、格式规范、AI 工具章程及原件哈希。
- `01_资料检索/`：采用与排除的文献证据；全文副本统一存放在 `原始文献/`。
- `02_数据与参数/`：附件审计、派生数据和可复现重建程序。
- `03_机理与推导/`：系统边界、模型选择、变量参数及实现映射。
- `问题/问题1` 至 `问题/问题4`：各问分析、推导、实现说明、验证、结论、源码和正式结果。
- `公共代码/`：几何距离、随机微构体、混合介质、Pareto 连通、结果注册和绘图样式模块。
- `tests/`、`benchmarks/`：回归测试与性能/实现路线验证。
- `论文/`：LaTeX 主文件、参考文献、生成内容、正式图件、FreeCAD 模型和审阅 PDF。`main.tex` + `content.tex` + `results.tex` + `source_appendix.tex` 构成完整论文源码。
- `审计/`：项目、论文、图件和交付门禁报告。

## 单一事实来源

- 参数、随机种子和问题依赖：`项目配置.json`
- 产物能力与稳定路径：`工作流产物清单.json`
- 阶段状态与门禁证据：`项目状态.json`
- 论文关键数值：`结果注册表.json`
- 结论与验证对应关系：`结论与证据记录.md`
- 图件来源、生成程序和审核：`图表设计与来源.md`、`论文/图表审核记录.md`
- 支撑材料白名单：`支撑材料配置.json`

同一数值只在结果注册表中冻结，再导出到 LaTeX；正文、图注和结果表不得分别手工维护副本。

## 复现与检查

项目级依赖顺序由 `run_pipeline.py` 管理；高成本正式运行的完整参数以
`论文/src/fragments/reproduction_commands.md` 为准。先用只读命令检查执行顺序：

```powershell
python run_pipeline.py --from-step 1 --to-step 4 --dry-run
python -m pytest -q
.\工作流.ps1 doctor
.\工作流.ps1 audit
```

完整稿与正文便阅稿的构建命令分别为：

```powershell
python 论文/src/build_latex.py --mode internal
python 论文/src/build_latex.py --mode internal --review
```

问题四确认、独立审计、结果回填和成本前沿图均已完成。正式提交构建仍需提供实际参赛组别与
`CM` 加 7 位数字编号；源码冻结、正式 PDF 和支撑材料命令见 `论文/src/README.md`。

## 清理规则

- `.pytest_cache/`、`__pycache__/`、`*.pyc`、LaTeX 辅助文件均可通过 `.gitignore` 排除。
- `问题/问题4/results/D_screen2000_confirm50000/` 整棵保留，以便逐分片复核正式结论。
- 各问不保留空置的 `data/`、`figures/`、`models/` 占位目录；正式数据归入 `results/`，正式图件与三维模型统一归入 `论文/figures/`。
