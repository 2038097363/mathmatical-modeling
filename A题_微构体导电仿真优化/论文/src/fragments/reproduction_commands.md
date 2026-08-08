#### B.1 软件与运行环境

| 软件 | 实际版本 | 用途 |
|---|---|---|
| Python | 3.13.5；Q4 确认进程为 3.13.7 | 几何计算、蒙特卡洛仿真、统计确认与绘图 |
| NumPy / SciPy / Numba | 2.1.3 / 1.15.3 / 0.61.0 | 数值计算、统计分布与几何核加速 |
| pandas / Matplotlib / openpyxl | 2.2.3 / 3.10.0 / 3.1.5 | 结果整理、科学绘图与附件读取 |
| FreeCAD | 1.1.1，Revision 20260414 | 参数化三维建模与视图导出 |
| Pandoc | 2.12 | Markdown 正文转换为 LaTeX 中间件 |
| WPS Office | KWPS.Application 12.0 | 官方 DOCX 模板的只读版式对照 |
| Poppler | 25.02.0 | PDF 页数核验与逐页栅格化 |

所有命令均从项目根目录运行。Python 依赖由下列命令安装：

```powershell
python -m pip install -r requirements.txt
```

#### B.2 数值结果复现

问题一及附件审计：

```powershell
python "02_数据与参数/src/reconstruct_segments.py"
python "问题/问题1/src/solve.py"
```

问题二主边界和两组边界敏感性试验：

```powershell
python "问题/问题2/src/solve.py" --boundary-mode D --max-count 720 --trials 20000 --seed 20260801 --stream-id 2 --workers 8 --batch-size 20 --counts 354 424 495 707 --output-dir "问题/问题2/results/D_primary_n20000" --resume --register-results
python "问题/问题2/src/solve.py" --boundary-mode B --max-count 720 --trials 5000 --seed 20260801 --stream-id 21 --workers 8 --batch-size 50 --counts 354 424 495 707 --output-dir "问题/问题2/results/B_sensitivity_n5000" --resume
python "问题/问题2/src/solve.py" --boundary-mode A --max-count 720 --trials 5000 --seed 20260801 --stream-id 22 --workers 8 --batch-size 50 --counts 354 424 495 707 --output-dir "问题/问题2/results/A_sensitivity_n5000" --resume
```

问题三独立确认：

```powershell
python "问题/问题3/src/solve.py" --threshold-artifact "问题/问题2/results/D_primary_n20000/threshold_samples.json" --output-dir "问题/问题3/results/D_confirmation_n50000" --confirmation-trials 50000 --confirmation-stream-id 3 --workers 8 --batch-size 100 --resume --register-results
```

问题四先执行完整筛选与冻结，再从冻结文件执行独立确认；命令本身不写入任何预设最终配比：

```powershell
python "问题/问题4/src/solve.py" --stage screen --output-dir "问题/问题4/results/D_screen2000_confirm50000" --target 0.90 --screening-trials 2000 --confirmation-trials 50000 --screening-candidate-rule cp_lower --familywise-confidence 0.95 --max-na 720 --max-nb 6000 --step-na 120 --step-nb 1000 --max-screening-designs 500 --seed 20260801 --screening-stream-id 4 --confirmation-stream-id 5 --workers 16 --screening-batch-size 10 --confirmation-batch-size 100 --resume
python "问题/问题4/src/solve.py" --stage freeze --output-dir "问题/问题4/results/D_screen2000_confirm50000" --screening-candidate-rule cp_lower --familywise-confidence 0.95 --seed 20260801 --screening-stream-id 4 --confirmation-stream-id 5
python "问题/问题4/src/solve.py" --stage confirm --output-dir "问题/问题4/results/D_screen2000_confirm50000" --target 0.90 --screening-trials 2000 --confirmation-trials 50000 --familywise-confidence 0.95 --seed 20260801 --screening-stream-id 4 --confirmation-stream-id 5 --workers 16 --confirmation-batch-size 100 --resume --register-results
```

#### B.3 测试、绘图与三维模型复现

```powershell
python -m pytest -q
python "论文/figures/src/build_q1_scenes.py"
python "论文/figures/src/build_q2_q3_figure.py"
python "论文/figures/src/build_q4_frontier.py" --screening "问题/问题4/results/D_screen2000_confirm50000/q4_screening.json" --final "问题/问题4/results/D_screen2000_confirm50000/q4_summary.json"
python "论文/figures/src/build_q4_mixed_scene.py" --design-json "问题/问题4/results/D_screen2000_confirm50000/q4_summary.json"
```

附录 C 的程序由显式白名单逐文件读取，标题下显示项目相对路径；构建清单同时记录每个源文件的 SHA-256，以便确认论文代码与实际执行代码一致。
