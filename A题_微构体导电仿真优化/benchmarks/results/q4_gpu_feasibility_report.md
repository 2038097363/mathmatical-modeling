# Q4 GPU 旁路可行性基准

## 结论

不启动 GPU 大规模确认运行。RTX 5060 Laptop GPU 可正确加速 A-B 完整球闭式距离批次，但在已完成 trial 2 上，乐观端到端加速仅为 1.087 倍，未达到预设的 2 倍门槛。

## 环境与隔离

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，显存 8151 MiB。
- Torch：2.11.0+cu128，CUDA 12.8，可正常使用 GPU。
- Numba：0.61.0，`numba.cuda.is_available()` 为 `False`。
- CPU 捕获、CUDA 批处理和 CPU 回放采用三个独立进程，避免 Torch 与 SciPy/Numba 的 OpenMP 运行库冲突。
- 仅新增 `benchmarks/` 和 `tmp/` 下的基准产物；未修改、停止或重启冻结确认流，未写入其 `confirmation/shards` 目录。

## 热点证据

冻结分片 `shard_000000_000099.json` 的平均耗时为 17.8146 s/trial。对 trial 0 的只读 `cProfile` 显示：总计时 32.2123 s，其中距离界/GJK 累计约 15.212 s，Pareto 扫描约 5.269 s；剖析会放大函数调用开销。对 trial 1 的低扰动分段计时为 22.8402 s，其中 Pareto 扫描 2.7067 s，建图及几何接触 20.1335 s。

## 逐 trial 等价性

基准使用冻结确认流已经完成的 trial 2：

- CPU 重算的 `connectivity_frontier` 与全部诊断字段和原 shard 逐键一致。
- 捕获 83,931 个真实 A-B 完整球候选，Torch CUDA float64 的宽相排除、接触判定和窄相调用数均为 0 差异。
- 将 CUDA 判定回放到完整接触图后，`connectivity_frontier`、几何诊断和 `pareto_search` 再次与原 shard 逐键一致。

## 性能

| 项目 | 时间 |
|---|---:|
| 冻结分片 CPU 均值 | 17.8146 s/trial |
| A-B 数组打包 | 0.2667 s |
| CUDA 批处理，含 H2D/D2H | 0.00450 s |
| CUDA 判定回放后的完整 trial | 16.1176 s |
| 乐观混合路径合计 | 16.3888 s |
| 乐观端到端加速 | 1.087x |

这里的 1.087 倍仍是乐观估计，因为数组打包基于已经捕获的候选，未计入正式重构、批次调度和异常回退的全部开销。

## 风险与停止规则

继续提高 GPU 覆盖率必须重写 A-A 平底圆柱 GJK、截球 active-set 支持映射、认证距离上下界、近阈值回退和诊断计数。此路径具有分支发散、消费级 GPU FP64 吞吐较低、阈值附近浮点判定变化及诊断不一致风险。预计原型和至少 100 个已完成 trial 的逐 trial 审计需要 8--16 小时，仍无明确超过 2 倍的端到端收益。因此按预设停止规则保留现有 CPU 冻结确认流。

机器可读证据见 `q4_torch_cuda_ab_feasibility.json`，复现实验入口为 `../benchmark_q4_torch_cuda_ab.py`。
