# Q1 正式逐行结论与历史诊断报告

> **正式口径：仅 `A_row_literal / disconnected_fragments`。** 组委会已确认附件每行是独立截断介质；下文 B/C 场景和 `connected_same_particle` 只保留为澄清前的机器诊断，不是本题答案，也不进入结果注册表。

## 计算口径

- 每条 Excel 记录按给定端点直接构造半径 30 nm 的有限平底圆柱；短段不拉长、不补段。
- 圆柱对先用胶囊距离作下界宽相；只有下界不排除 1.8 nm 接触时才调用平底圆柱 GJK 窄相。
- 只使用基础域内记录的直接欧氏距离；没有对任意两行使用全局 minimum-image。
- 电极 x=±5000 nm 采用圆柱投影区间的闭式距离。
- 正式 A 为逐行实体；历史诊断 B 为三组均按 10000 立方体接头映射；历史诊断 C 为组1/2按 10000×1000×1000 候选薄域、组3按 10000 立方体映射。
- 历史诊断 `connected_same_particle` 只加入 `junctions.csv` 中一对一、共线的候选周期接头；残差不闭合的链不补段，但其已有唯一接头仍可保持内部导通。正式 `disconnected_fragments` 不加入任何此类内部边。

## 几何筛选审计

|组|记录数|总对数|胶囊排除|GJK窄相|接触|窄相分离|不确定|电极边|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|组1|12|66|64|2|2|0|0|7|
|组2|49|1176|1149|27|27|0|0|22|
|组3|535|142845|142679|166|165|1|0|182|

全部窄相对另用六变量凸约束 SLSQP 独立复核；该检查不替代 GJK 上下界证书。三组阈值分类不一致总数为 0，最大距离估计差为 1.054e-07 nm。
GJK 中共有 6 对未触发收敛标志；所有窄相上下界最大宽度为 6.046e-09 nm，距 1.8 nm 阈值的最小确定裕量为 0.021329 nm，故没有分类不确定边。

## 导通结果

|解释|内部连接|组|行数|particle数|接头数|结论|行号见证路径|
|---|---|---|---:|---:|---:|---|---|
|A_row_literal|disconnected_fragments|组1|12|12|0|nonconductive|-|
|A_row_literal|disconnected_fragments|组2|49|49|0|conductive|LEFT -> r13 -> r14 -> r26 -> r41 -> RIGHT|
|A_row_literal|disconnected_fragments|组3|535|535|0|conductive|LEFT -> r65 -> r266 -> r218 -> r353 -> RIGHT|
|A_row_literal|connected_same_particle|组1|12|12|0|nonconductive|-|
|A_row_literal|connected_same_particle|组2|49|49|0|conductive|LEFT -> r13 -> r14 -> r26 -> r41 -> RIGHT|
|A_row_literal|connected_same_particle|组3|535|535|0|conductive|LEFT -> r65 -> r266 -> r218 -> r353 -> RIGHT|
|B_full_cube_periodic|disconnected_fragments|组1|12|9|3|nonconductive|-|
|B_full_cube_periodic|disconnected_fragments|组2|49|39|10|conductive|LEFT -> r13 -> r14 -> r26 -> r41 -> RIGHT|
|B_full_cube_periodic|disconnected_fragments|组3|535|357|178|conductive|LEFT -> r65 -> r266 -> r218 -> r353 -> RIGHT|
|B_full_cube_periodic|connected_same_particle|组1|12|9|3|conductive|LEFT -> r4 -> r12 -> RIGHT|
|B_full_cube_periodic|connected_same_particle|组2|49|39|10|conductive|LEFT -> r12 -> r41 -> RIGHT|
|B_full_cube_periodic|connected_same_particle|组3|535|357|178|conductive|LEFT -> r10 -> r452 -> RIGHT|
|C_thin_12_full_3|disconnected_fragments|组1|12|7|5|nonconductive|-|
|C_thin_12_full_3|disconnected_fragments|组2|49|28|21|conductive|LEFT -> r13 -> r14 -> r26 -> r41 -> RIGHT|
|C_thin_12_full_3|disconnected_fragments|组3|535|357|178|conductive|LEFT -> r65 -> r266 -> r218 -> r353 -> RIGHT|
|C_thin_12_full_3|connected_same_particle|组1|12|7|5|conductive|LEFT -> r4 -> r12 -> RIGHT|
|C_thin_12_full_3|connected_same_particle|组2|49|28|21|conductive|LEFT -> r11 -> r50 -> RIGHT|
|C_thin_12_full_3|connected_same_particle|组3|535|357|178|conductive|LEFT -> r10 -> r452 -> RIGHT|

## 粒子/行号见证路径

- `A_row_literal / disconnected_fragments / 组1`：-
- `A_row_literal / disconnected_fragments / 组2`：LEFT -> 组2_row_literal_P0011[r13;cannot_reconstruct] -> 组2_row_literal_P0012[r14;unique] -> 组2_row_literal_P0024[r26;unique] -> 组2_row_literal_P0039[r41;cannot_reconstruct] -> RIGHT
- `A_row_literal / disconnected_fragments / 组3`：LEFT -> 组3_row_literal_P0063[r65;cannot_reconstruct] -> 组3_row_literal_P0264[r266;unique] -> 组3_row_literal_P0216[r218;unique] -> 组3_row_literal_P0351[r353;cannot_reconstruct] -> RIGHT
- `A_row_literal / connected_same_particle / 组1`：-
- `A_row_literal / connected_same_particle / 组2`：LEFT -> 组2_row_literal_P0011[r13;cannot_reconstruct] -> 组2_row_literal_P0012[r14;unique] -> 组2_row_literal_P0024[r26;unique] -> 组2_row_literal_P0039[r41;cannot_reconstruct] -> RIGHT
- `A_row_literal / connected_same_particle / 组3`：LEFT -> 组3_row_literal_P0063[r65;cannot_reconstruct] -> 组3_row_literal_P0264[r266;unique] -> 组3_row_literal_P0216[r218;unique] -> 组3_row_literal_P0351[r353;cannot_reconstruct] -> RIGHT
- `B_full_cube_periodic / disconnected_fragments / 组1`：-
- `B_full_cube_periodic / disconnected_fragments / 组2`：LEFT -> 组2_full_cube_periodic_P0011[r13;unique] -> 组2_full_cube_periodic_P0012[r14;unique] -> 组2_full_cube_periodic_P0024[r26;unique] -> 组2_full_cube_periodic_P0010[r41;unique] -> RIGHT
- `B_full_cube_periodic / disconnected_fragments / 组3`：LEFT -> 组3_full_cube_periodic_P0063[r65;cannot_reconstruct] -> 组3_full_cube_periodic_P0223[r266;unique] -> 组3_full_cube_periodic_P0191[r218;unique] -> 组3_full_cube_periodic_P0296[r353;cannot_reconstruct] -> RIGHT
- `B_full_cube_periodic / connected_same_particle / 组1`：LEFT -> 组1_full_cube_periodic_P0002[r4,r12;unique] -> RIGHT
- `B_full_cube_periodic / connected_same_particle / 组2`：LEFT -> 组2_full_cube_periodic_P0010[r12,r41;unique] -> RIGHT
- `B_full_cube_periodic / connected_same_particle / 组3`：LEFT -> 组3_full_cube_periodic_P0008[r10,r452;unique] -> RIGHT
- `C_thin_12_full_3 / disconnected_fragments / 组1`：-
- `C_thin_12_full_3 / disconnected_fragments / 组2`：LEFT -> 组2_thin_prism_periodic_P0011[r13;unique] -> 组2_thin_prism_periodic_P0012[r14;unique] -> 组2_thin_prism_periodic_P0021[r26;unique] -> 组2_thin_prism_periodic_P0010[r41;unique] -> RIGHT
- `C_thin_12_full_3 / disconnected_fragments / 组3`：LEFT -> 组3_full_cube_periodic_P0063[r65;cannot_reconstruct] -> 组3_full_cube_periodic_P0223[r266;unique] -> 组3_full_cube_periodic_P0191[r218;unique] -> 组3_full_cube_periodic_P0296[r353;cannot_reconstruct] -> RIGHT
- `C_thin_12_full_3 / connected_same_particle / 组1`：LEFT -> 组1_thin_prism_periodic_P0002[r4,r12;unique] -> RIGHT
- `C_thin_12_full_3 / connected_same_particle / 组2`：LEFT -> 组2_thin_prism_periodic_P0009[r11,r50;unique] -> RIGHT
- `C_thin_12_full_3 / connected_same_particle / 组3`：LEFT -> 组3_full_cube_periodic_P0008[r10,r452;unique] -> RIGHT

## 边证据说明

- `physical_contact_edges.csv` 给出全部确定物理接触边的最小间隙上下界和 1.8 nm 关系。
- `narrow_phase_pairs.csv` 还保留胶囊未能排除、经 GJK 判为分离或不确定的全部近邻对。
- `independent_slsqp_validation.csv` 给出全部窄相对的独立凸优化距离检查及阈值分类一致性。
- `scenario_graph_edges.csv` 给出每种解释/内部模式实际使用的全部图边及两端 particle_id。
- `witness_path_edges.csv` 是每条导通见证路径的逐边证据。周期内部边不以 1.8 nm 判定，间隙列为空，另报映射后端点残差。
- `internal_junction_audit.csv` 同时列出启用与禁用模式下每个接头，便于核对未闭合链没有被补段。

## 可复现性

- Python：`3.13.5`；NumPy：`2.1.3`；openpyxl：`3.1.5`。
- 附件 SHA-256：`6DC68DD49356AEAB483906A524FF79855B454FCACCD168F81FF3A3AD989C4C51`。
- 几何核 SHA-256：`652CB279E662B083E190885557A942B9F819A789E77D9C538945716C91027E32`。
- 运行耗时：`9.305 s`。

```powershell
cd C:\Users\23258\Desktop\华数杯\A题_微构体导电仿真优化
python -m unittest discover -s tests -v
python 问题\问题1\src\solve.py
```
