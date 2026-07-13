# 三堆N+1退化感知功率控制方法定义

更新时间：2026-07-13

本文档固定当前论文主方法的符号、方程、实现映射和解释边界。当前范围只含三个PEMFC电堆之间的
运行堆选择与功率分配；不含锂电池外层，也不使用未来需求功率。

## 1. 系统与状态

电堆集合为 $\mathcal{S}=\{0,1,2\}$。每个快层时刻 $t$ 的系统状态为

$$
x_t=\{D_{j,t}, I_{j,t-1}, z_{j,t-1}, \tau_{j,t}\}_{j\in\mathcal{S}},
$$

其中 $D_{j,t}\ge 0$ 是累计退化指标，$I_{j,t-1}$ 是上一时刻电流，$z_{j,t-1}\in\{0,1\}$
表示电堆是否激活，$\tau_{j,t}$ 是离散动作驻留时间。$D$使用文献系数定义的百分比型代理单位，
不是已校准的真实SOH。

快层动作是 $a_t=\{I_{j,t},z_{j,t}\}_{j\in\mathcal{S}}$，电流来自冻结离散集合

$$
\mathcal{I}=\{0,25,90,120,160,195,270,370\}\ {\rm A}.
$$

当需求为正时恰有两堆激活，即 $\sum_j z_{j,t}=2$；第三堆休息。慢层给定在线集合
$\mathcal{A}_k\subset\mathcal{S}$ 后，快层强制 $z_{j,t}=0,\forall j\notin\mathcal{A}_k$。
动作网格、两堆在线和功率跟踪作为硬约束。最小驻留优先作为硬约束；若其导致当前需求没有
可行动作，控制器允许有记录的安全覆盖，并单独统计覆盖步数。

实现：`src/fc_power/world_model/mechanistic.py`、
`src/fc_power/power_allocation/multistack_allocator.py`。

## 2. 实车负载映射

实车单堆测量功率为 $P_t^{\rm real}$。当前冻结开发参考 $P_{\rm norm}=30$ kW，系统负载为

$$
\ell_t={\rm clip}\!\left(\frac{P_t^{\rm real}}{P_{\rm norm}},0,1\right),\qquad
P_t^{\rm dem}=\ell_t(1-\rho)P_{2,0}^{\max},
$$

其中 $\rho=0.05$ 是容量余量，$P_{2,0}^{\max}$ 是健康状态下两台电堆的聚合参考功率。
标定段为segment 0-21，留出段为22-45，按完整segment间最大时间缺口切分；经验转移不跨停机区
或segment边界。

30 kW只来自标定期控制目标上限，不是物理额定值。留出期11.59%的正功率样本高于该参考；
不截峰且取最坏健康/策略组合时，6.65%的留出正功率步超过双堆物理容量。36.756 kW是使用
留出峰值反算的事后严格容量下界，不能回填标定；40 kW仅是待外部资料确认的控制目标候选。

实现：`scripts/26_audit_zuo_real_load_calibration.py`、
`scripts/39_build_service_exposure_templates.py`、`scripts/45_audit_holdout_capacity_shift.py`。

## 3. 动作驱动健康转移

对电堆 $j$，一步条件均值退化写为

$$
\begin{aligned}
\Delta \bar D_{j,t}={}&
\eta_j r(I_{j,t},z_{j,t})\frac{\Delta t}{3600}
+\eta_j r_{\rm nat}z_{j,t}\frac{\Delta t}{3600}\\
&+\eta_j k_{\rm ramp}|I_{j,t}-I_{j,t-1}|
+\eta_j k_{\rm shift}\mathbf{1}_{\rm shift}\\
&+\eta_j k_{\rm start}\mathbf{1}_{0\rightarrow1}
+\eta_j k_{\rm stop}\mathbf{1}_{1\rightarrow0},
\end{aligned}
$$

其中 $\eta_j$ 是堆间异质性因子。当前实现把自然运行项并入电流速率表 $r(I,z)$，因此
$r_{\rm nat}=0$，不会重复相加。冻结文献系数为：启动$0.00196$%/次、高载
$0.001470$%/h、低载$0.00126$%/h、激活自然退化$0.002$%/h、变载
$5.93\times10^{-5}$%/次；当前斜率项和停机损伤为零。所有分量非负，因此

$$
D_{j,t+1}=D_{j,t}+\Delta D_{j,t}\ge D_{j,t}.
$$

快层控制和确定性主基线使用 $\Delta D_{j,t}=\Delta\bar D_{j,t}$。Gamma随机性只作用于连续负载
分量：若其均值为 $\mu>0$、尺度为 $\beta$，则

$$
\Delta D_{j,t}^{\rm load}\sim {\rm Gamma}(\mu/\beta,\beta),
\quad \mathbb{E}[\Delta D_{j,t}^{\rm load}]=\mu.
$$

单秒shape约为$10^{-3}$量级，不能用短轨迹样本均值排序策略；因此Gamma只进入小时级聚合暴露
和离线风险消融，不进入秒级主控制采样。

实现：`src/fc_power/health/gamma_process.py`、
`src/fc_power/health/lzw_gamma_calibration.py`。

## 4. 健康到IV/功率闭环

LZW 6,104事件轨迹给出 $\theta=[i_0,i_h,R_{\rm ohm}]$。固定文献系数先把动作暴露映射为
$D$，再拟合单调幂律

$$
\theta_m(D)=\theta_{m,0}+(\theta_{m,1}-\theta_{m,0})
\left(\frac{D}{D_{\rm ref}}\right)^{p_m},
$$

其中 $D_{\rm ref}=9.3530285$%，三个指数分别为2.0493、0.7408和1.6039。随后使用LZW
半经验IV模型计算 $V_j(I_{j,t},\theta_j(D_{j,t+1}))$，并得到

$$
P_{j,t}=N_{\rm cell} I_{j,t}V_j/1000.
$$

因此每个实际执行动作都会先更新 $D$，再更新 $\theta$、IV曲线和下一步可用功率；健康不是固定
标签。该映射是跨数据链退化代理，尚未证明与21UBE0022逐行同车同堆。

实现：`src/fc_power/health/dynamic_proxy.py`、`src/fc_power/lzw_iv_model.py`、
`src/fc_power/world_model/mechanistic.py`。

## 5. 秒级快层

在当前状态和当前需求下，Instant快层枚举满足硬约束的动作，并求

$$
a_t^*=\arg\min_{a\in\mathcal{F}(x_t,P_t^{\rm dem},\mathcal{A}_k)}
\left(w_Hc_H+w_Dc_D+w_\theta c_\theta+w_Pc_P+w_Sc_S+w_Rc_R\right).
$$

$c_H$为归一化氢耗，$c_D$为一步退化增量，$c_\theta$为健康相关性能损失，$c_P$为FC跟踪误差，
$c_S$和$c_R$分别为开关和电流跃迁。当前权重为0.45、1.0、0.25、5.0、0.08和0.005。
FC-only模式满足

$$
\left|\sum_jP_{j,t}-P_t^{\rm dem}\right|\le 5.5\ {\rm kW}.
$$

控制只读取当前需求，不构造未来预览。跟踪候选剪枝使用与世界模型相同的确定性健康转移和IV
功率，测试已证明与原穷举动作、目标和约束一致。

## 6. 24小时慢层

慢层每24小时最多选择两堆运行。主策略不求复杂随机优化，而是选择当前累计退化最小的两堆：

$$
\mathcal{A}_k^{\rm HG}=\underset{\mathcal{A}\subset\mathcal{S},|\mathcal{A}|=2}
{\arg\min}\ \sum_{j\in\mathcal{A}}D_{j,k},
$$

等价于让当前最老堆休息。对选中的无序集合，再根据开发模板的两个快层角色暴露
$e_0,e_1$选择角色顺序

$$
\pi_k^*=\arg\min_{\pi\in\{(a,b),(b,a)\}}
\max_{r\in\{0,1\}}\left(D_{\pi(r),k}+\eta_{\pi(r)}e_r\right).
$$

该决策只使用当前健康和开发模板均值，不读取未来需求。Expected-max和Gamma-CVaR枚举六个有序
分配，但在强health-greedy基线上仅有很小且不稳定的附加收益，故只保留为消融。

实现：`src/fc_power/evaluation/service_scheduler.py`、
`scripts/36_run_service_horizon_scheduler.py`。

## 7. 评价量与统计设计

长期主评价是到LZW标定健康边界的时间

$$
T_B=\inf\{t:\max_jD_j(t)\ge D_{\rm ref}\}.
$$

$D_{\rm ref}$不是已验证失效阈值，因此 $T_B$只能称“健康边界到达时间”，不能称真实寿命或RUL。
策略比较使用同负载种子、同健康随机数的配对统计；开发阶段使用120条快层模板、20个种子和
单因素稳健性，最终参数冻结后才读取segment 22-45。

完整留出验证覆盖86,415个1秒样本。每个完整segment内部连续携带动作、驻留和健康，段间不跨
未知时间缺口。3种健康身份、24段、2策略共144例和518,490步；约束违规为0，另有455个
有审计驻留安全覆盖步（0.088%）。不把独立段拼成虚构长行程。最长留出segment为7.543 h，
因此每段只在入口执行一次冻结慢层分配；这验证入口选择和段内快层，不验证24 h动态重复调度。

终端最大退化策略差以8个完整正功率segment为统计单位。两个预声明非平凡健康身份使用单侧
精确Wilcoxon配对检验并做Holm校正，同时用10,000次segment bootstrap给出均值差的95%区间；
不把518,490个相关逐秒步当成独立样本。

## 8. 当前不可声称内容

- 不称30 kW为物理额定功率，不称截峰回放为原始峰值全保真验证；
- 不称 $D_{\rm ref}$为失效阈值，不称边界时间为真实寿命；
- 不称Gamma单秒采样改善控制，主方法秒级使用条件均值；
- 不称健康状态是实车posterior，当前只有动作驱动预测态；
- 不称方法同时优化了电池或使用了未来功率预测；
- 不称health-greedy对全部指标Pareto占优，它降低最差堆健康代价但总期望退化可略升。
