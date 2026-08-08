# 2026 年华数杯数学建模竞赛

第七届"华数杯"大学生数学建模竞赛参赛项目，选题为 **A 题：微构体中填充导电介质的仿真优化**。

## 项目结构

```
华数杯/
├── README.md                                    # 本文件
├── .gitignore                                   # 仓库级忽略规则
├── 2026年第七届华数杯数学建模竞赛赛题/            # 竞赛原始赛题（A/B/C 题）
├── 绘图模板/                                     # 论文图形设计参考模板
└── A题_微构体导电仿真优化/                        # A 题求解项目
    ├── README.md                                # 项目详细说明
    ├── 00_赛题与附件/                            # 题面、附件与竞赛规范
    ├── 01_资料检索/                              # 文献检索与原始文献
    ├── 02_数据与参数/                            # 数据重建与参数审计
    ├── 03_机理与推导/                            # 系统边界、模型选择与变量映射
    ├── 04_验证与拓展/                            # 验证计划与拓展分析
    ├── 问题/                                     # 各问求解
    │   ├── 问题1/  # 几何接触与连通判定
    │   ├── 问题2/  # Monte Carlo 渗流概率
    │   ├── 问题3/  # 精确置信界与最小填充量
    │   └── 问题4/  # Pareto 前沿与成本优化
    ├── 公共代码/                                 # 共享求解模块
    ├── 论文/                                     # LaTeX 论文源码与 PDF
    ├── tests/                                    # 回归测试
    ├── benchmarks/                               # 性能基准
    └── 审计/                                     # 项目与论文审计报告
```

## 论文

- **完整稿**：[A题论文.pdf](A题_微构体导电仿真优化/论文/A题论文.pdf) — 含 24 个程序的源码附录
- **便阅稿**：[A题论文_便阅稿.pdf](A题_微构体导电仿真优化/论文/A题论文_便阅稿.pdf) — 省略源码附录

## 快速开始

### 环境要求

```powershell
pip install -r A题_微构体导电仿真优化/requirements.txt
```

### 运行测试

```powershell
cd A题_微构体导电仿真优化
python -m pytest -q
```

### 构建论文

```powershell
cd A题_微构体导电仿真优化
python 论文/src/build_latex.py --mode internal          # 完整稿
python 论文/src/build_latex.py --mode internal --review  # 便阅稿
```

需要 XeLaTeX + latexmk + pandoc。

### 复现求解

```powershell
cd A题_微构体导电仿真优化
python run_pipeline.py --from-step 1 --to-step 4 --dry-run
python run_pipeline.py --from-step 1 --to-step 4
```

## 关键技术

- **几何计算**：平底圆柱 GJK 距离、支持映射、胶囊边界
- **随机模拟**：Monte Carlo 固定种子、共同前缀、增量并查集
- **统计推断**：Clopper-Pearson 精确置信界、Wilson 区间、Bonferroni 联合校正
- **优化**：二维 minimax Pareto 前沿、成本有界整数域扫描
- **可视化**：FreeCAD 三维建模、Matplotlib 科学图表

## 许可证

本项目仅用于学术交流与竞赛展示。竞赛赛题版权归属华数杯组委会。
