# 论文规范证据表

本目录由`scripts/46_build_paper_evidence_tables.py`从冻结实验结果自动生成。

- `table01`：方法参数与解释边界；
- `table02`：四种负载场景的长期强基线；
- `table03`：11组单因素稳健性；
- `table04`：冻结窗口oracle选择；
- `table05`：全部真实留出段可行性和代价权衡；
- `table06`：归一化参考与N+1容量审计；
- `table07`：完整留出segment bootstrap点估计与95%区间；
- `table08`：预声明最差堆主检验与Holm校正；
- `table09-10`：逐段删一与单段符号反转影响力审计；
- `table11-13`：30/35/40 kW事后归一化敏感性；
- `table14`：冻结最大误差案例的跟踪容差边界；
- `table15-16`：12个慢层决策点的自身目标遗憾和分配明细；
- `claim_values.json`：正文可引用的规范数值；
- `source_manifest.json`：每个输入文件的SHA-256。

任何正文数字应先进入`claim_values.json`，不得从图上估读或手工改写。40 kW仍是待物理资料
确认的候选，不因出现在规范表中而成为已验证额定值。
