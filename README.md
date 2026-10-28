# cnt-water-nacl-md
Molecular dynamics simulations of water and NaCl in carbon nanotubes: methods, scripts, progress, and results.


## 本地整理范围

本仓库保存 CNT 纯水与 NaCl 分子动力学研究的方法、进展、脚本与参数。原始模拟数据保留在仓库外。

- [研究方法](docs/methods/README.md)
- [进展记录](docs/progress/2026-10-28.md)
- [精选结果建议](docs/results-selection.md)（结果文件尚未复制）
- [来源及 SHA256 清单](docs/source-manifest.tsv)
- `scripts/pure_water/`、`scripts/nacl/`：首批精选源码
- `configs/`：CNT(6,6) 纯水与 1 M NaCl 的代表性配置

## 使用范围

这些脚本是现有研究源码的归档副本，未执行模拟或分析。部分包含 `/scratch/yingyiz94/` 绝对路径，并可能创建或覆盖原目录中的文件。运行前必须检查输入、输出、依赖和路径；不要直接运行 setup 脚本。

本次只精选了部分源码，不是完整可移植工作流。运行所需结构、轨迹、检查点、部分汇总输入及力场依赖未随仓库收录；配置示例不能单独重建体系。拓扑引用的外部文件及其授权需在扩展复现材料时确认。

保留远程原始 README 内容、忽略规则及提交历史；当前整理尚未提交或推送。
