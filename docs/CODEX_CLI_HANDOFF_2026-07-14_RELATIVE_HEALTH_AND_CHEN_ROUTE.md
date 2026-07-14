# Codex CLI 交接：相对健康变通与陈鹏效率主线

更新时间：2026-07-14
分支：`codex/gamma-health-foundation`
工作起点：`bf94f3b`
负载与健康基础提交：`d11f027`

## 当前结论

- 动作分辨的真实退化模型仍为 NO-GO；现有数据不能可靠分离时间、负载、启停和环境影响。
- 新的系数无关变通为 `LIMITED_GO_MECHANISM_ONLY`，只允许解释为相对健康预算重分配机制。
- 主试验固定各策略相同的全车健康进度预算，按实际堆输出份额在线更新 `h`。
- 相对在线性能自适应，相对健康策略降低最终最大 `h` `0.012217`，块 bootstrap 95% 区间 `[-0.014793,-0.009643]`；跟踪 MAE 增加 `0.0211 kW`。
- 去掉瞬时过供电后，有效电效率只变化 `+0.0006` 个百分点且区间跨0，不作为效率改善证据。
- 该改善部分来自目标函数显式偏好较小 `h`，不能当作真实退化算法的独立有效性证明。
- 推荐把陈鹏效率优化升级成受约束的混合离散-连续动态分配主线；相对健康层只作可选消融。

## 关键入口

- 总结与路线：`docs/RELATIVE_HEALTH_WORKAROUND_AND_CHEN_PAPER_ROUTE_2026-07-14.md`
- 分配器：`src/fc_power/power_allocation/relative_health_allocator.py`
- 实验：`scripts/73_run_relative_health_adaptive_replay.py`
- 测试：`tests/test_relative_health_allocator.py`
- 结果：`data/results/relative_health_adaptive_replay/`
- 图：`data/results/figures/fc_only_foundation/fig35_relative_health_adaptive_replay.png`

## 复现

```powershell
cd 'H:\其他\2026刘展玮\FC'
$env:PYTHONPATH='src'
python scripts\73_run_relative_health_adaptive_replay.py
python -m unittest tests.test_relative_health_allocator -v
```

完整回放在当前机器约需 310 秒。源文件 SHA256：

- 外部负载块：`2FA0E0519A3A338CFD89D5460E32EC6ADF156EFAE7264749461BFB7FA7AB6F45`
- LZW `h -> theta`：`DF383817ED48D7948D61EED645F8198F6F29ADF6357365C9EE9E95F4BAA7976E`
- IV 条件：`623B189FF62CCA6BDD88502AA1994D4993FEA17D55226B1048187EFD1AA47545`

## 下一步唯一主任务

先恢复陈鹏净效率模型的独立可运行性，统一净/毛功率口径并恢复空压机安全约束，再用精确枚举/SQP复现原效率曲线和功率分配结果。该基准没有通过前，不训练 CHDP，也不增加电池或未来需求预测。
