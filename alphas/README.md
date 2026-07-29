# Alpha 文件安全说明

本目录现有 `alpha_vwap5_*.parquet` 均由未来 VWAP 收益构造，含前视信息，只能用于
流程验证、性能标定和测试，禁止用于正式研究、实盘或对外披露。

使用这些文件时必须保留：

```yaml
alpha:
  source: file
  path: "alphas/alpha_vwap5_xxx.parquet"
  synthetic: true
```

`--alpha-file` 只覆盖文件路径，不会自动把 `synthetic` 改为 `false`。真实、PIT 可得的
生产 Alpha 应存放在有数据来源和可得日说明的独立位置，并在复核后显式配置
`synthetic: false`。
