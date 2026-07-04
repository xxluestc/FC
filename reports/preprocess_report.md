# Phase 2 预处理报告

预处理脚本：`scripts/01_preprocess_data.py`。

规则：

1. 只读取刘占伟 21UBE0022 半月原始CSV，不读取“故障数据”子目录；
2. UTF-8字段按语义重命名，不按易变的列号读取；
3. 时间戳解析失败行删除；同一时间戳多包记录对数值列取均值；
4. 相邻记录间隔大于10 s划为新 segment；
5. 每个 segment 内重采样到1 s，最多填补短间断；严禁跨停驶/跨夜插值；
6. 车速从 km/h 转 m/s，加速度仅在 segment 内差分；
7. FC、DCDC、电池和电机功率均保留原始符号，并由 V×I/1000 派生；符号和传感器缩放将在车辆动力学校准阶段审计，当前不擅自翻转。

输出 `data/processed/liu_vehicle_canonical_1s.csv` 是本地生成文件，不提交Git。具体行数和片段数由 `data/processed/preprocess_summary.json` 记录。

