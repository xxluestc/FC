# 代码框架说明

代码按“数据→动力学→预测→电堆/电池成本→滚动分配→评价”拆分。`src/`只含可复用函数和模型，`scripts/`负责阶段编排，`configs/`保存非敏感参数，`data/processed`和大结果默认不提交。

关键依赖关系：

```text
data.preprocess
  ├─> vehicle_dynamics
  └─> prediction
        ├─> speed_only_dynamics ─> vehicle_dynamics
        ├─> state_direct_power
        └─> hybrid_physics_corrected ─> vehicle_dynamics + residual

stack_model + hydrogen_model + degradation_cost + battery_model
  └─> power_allocation.mpc_allocator
        └─> evaluation
```

`scripts/04_train_or_run_predictors.py`采用严格时间切分：前70%训练、15%验证、后15%测试，窗口不跨运行断点。逐秒轨迹用于事件与RampRisk，专用窗口头用于平均功率/累计能量，二者不混称。

`scripts/05_run_power_allocation.py`在同一3600秒片段比较Instant、Constant、Perfect和Predicted；Constant/Perfect/Predicted均测试H=3/5/10。Perfect只作非因果上界。输出同时给出相对Instant与Constant的变化。
