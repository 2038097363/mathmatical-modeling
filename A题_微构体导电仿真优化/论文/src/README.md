# 论文构建与交付链

权威论文链为 Markdown 到 Pandoc 到 XeLaTeX PDF。官方 DOCX 模板只用于读取格式参数和版式
对照，不是正式论文源文件。所有命令均从项目根目录运行。

## 权威文件

- 正文源：`论文/内容稿.md`
- LaTeX 框架：`论文/main.tex`
- 构建器：`论文/src/build_latex.py`
- 生成中间件：`论文/generated/content.tex`、`results.tex` 和构建清单
- 完整审阅 PDF（附录列出四问真实主程序）：`论文/A题论文_当前审阅稿_LaTeX.pdf`
- 正文便阅 PDF（省略源码附录）：`论文/A题论文_正文便阅稿_LaTeX.pdf`
- 源码附录白名单：`论文/src/source_appendix_allowlist.json`
- 支撑材料白名单：`支撑材料配置.json`

`论文/generated/` 由脚本生成，不手工维护。数值先在 `结果注册表.json` 冻结，再进入正文与
图表；正式 Q4 结果不得直接写死在 `main.tex`。

## 内部审阅稿

生成正文、附录说明和四问主程序的完整内部稿；24 份求解、依赖、验证、绘图和 CAD 源码仍全部冻结到支撑材料：

```powershell
python 论文/src/build_latex.py --mode internal
```

只需快速检查正文排版时，可另生成省略源码附录的便阅稿：

```powershell
python 论文/src/build_latex.py --mode internal --review
```

两个文件名明确区分“完整稿”和“正文便阅稿”。两者均包含醒目的 INTERNAL-QA 标记，
在实际参赛组别与编号填写前不得提交。

## 正式 PDF

正式构建必须等待 Q4 确认、独立审计、前沿图和源码白名单全部冻结，并显式提供实际组别与
报名系统编号：

```powershell
python 论文/src/build_latex.py `
  --mode final `
  --group <专科生|本科生|研究生> `
  --competition-id <CM加7位数字>
```

正式输出文件名为 `A<CM编号>.pdf`。构建器会检查 Q4 证据哈希、匿名元数据、残留占位符和
题意处理过程叙述；任何一项不通过都会拒绝生成提交稿。

## 源码与支撑材料

Q4 冻结后先将 `source_appendix_allowlist.json` 的状态改为 `frozen` 并核对文件清单，再生成
匿名提交源码副本：

```powershell
python 论文/src/prepare_submission_sources.py
python 论文/src/prepare_support_materials.py --team-id <CM加7位数字>
```

源码副本只允许输出到此前不存在的 `提交源码/`，并记录源文件与副本的 SHA-256。支撑材料只
读取 `支撑材料配置.json` 白名单，不收录 `tmp/`、分片、缓存或论文旧稿。

## 验证

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_latex_chain.py tests/test_support_materials.py
.\工作流.ps1 audit
```

每次正文、图件或 Q4 数值变化后都重新构建 PDF，用 Poppler 渲染全部页面，并逐页检查摘要
首屏、页码、公式、表格、图注、裁切、空白和附录起页。

## DOCX 兼容工具

`build_docx.py`、`audit_docx.py`、`check_docx_readiness.py`、`render_docx.ps1` 及其映射文件仅
保留为官方模板对象、边距、字体和匿名属性的兼容审计工具。它们不会进入权威 LaTeX 构建或
最终支撑材料，除非竞赛平台后来明确改为要求 Word 提交。
