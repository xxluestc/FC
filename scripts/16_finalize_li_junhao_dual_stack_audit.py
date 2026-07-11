"""Refresh derived dual-stack quantities and write the research suitability report."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("experiments/li_junhao_dual_stack_audit")
PROCESSOR_PATH = Path(__file__).with_name("15_process_li_junhao_dual_stack.py")


def load_processor():
    spec = importlib.util.spec_from_file_location("li_processor", PROCESSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    processor = load_processor()
    processor.style()
    metric_rows = []
    hydrogen_rows = []
    for engine in processor.ENGINES:
        path = OUT / f"{engine}_three_day_processed.csv"
        frame = pd.read_csv(path, low_memory=False, parse_dates=["timestamp", "source_date"])
        ratio = frame["fc_voltage_V"] / (frame["stack_A_section_avg_V"] + frame["stack_B_section_avg_V"])
        effective_sections = ratio.replace([np.inf, -np.inf], np.nan).dropna().median()
        frame["effective_sections_per_stack"] = effective_sections
        voltage_sum = frame["stack_A_section_avg_V"] + frame["stack_B_section_avg_V"]
        frame["stack_A_voltage_V"] = frame["fc_voltage_V"] * frame["stack_A_section_avg_V"] / voltage_sum
        frame["stack_B_voltage_V"] = frame["fc_voltage_V"] * frame["stack_B_section_avg_V"] / voltage_sum
        frame["stack_A_power_kW_inferred"] = frame["stack_A_voltage_V"] * frame["fc_current_A"] / 1000
        frame["stack_B_power_kW_inferred"] = frame["stack_B_voltage_V"] * frame["fc_current_A"] / 1000
        frame["stack_power_sum_kW_inferred"] = frame["stack_A_power_kW_inferred"] + frame["stack_B_power_kW_inferred"]
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        selected_names = frame["source_file"].drop_duplicates().tolist()
        selected = [Path(name) for name in selected_names]
        metric = processor.metrics(frame, engine, selected)
        metric["effective_sections_per_stack"] = effective_sections
        metric_rows.append(metric)
        rate = frame["hydrogen_consumption_rate"].dropna()
        remaining = frame["hydrogen_remaining"].dropna()
        hydrogen_rows.append({
            "engine": engine, "rate_min": rate.min(), "rate_median": rate.median(), "rate_p95": rate.quantile(.95),
            "rate_max": rate.max(), "remaining_min": remaining.min(), "remaining_max": remaining.max(),
            "remaining_net_change": remaining.iloc[-1] - remaining.iloc[0] if len(remaining) else np.nan,
        })
        one_file = pd.read_csv(OUT / "selected_days.csv").query("engine == @engine and scope == 'one_day'")["file"].iloc[0]
        one_day = frame[frame["source_file"] == one_file].reset_index(drop=True)
        processor.plot_stack_power(one_day, engine, "1\u5929", OUT / "plots" / f"{engine}_1day_dual_stack_power.png")
        processor.plot_stack_power(frame, engine, "\u8fde\u7eed3\u5929", OUT / "plots" / f"{engine}_3day_dual_stack_power.png")
    metrics = pd.DataFrame(metric_rows)
    hydrogen = pd.DataFrame(hydrogen_rows)
    metrics.to_csv(OUT / "power_relationship_metrics.csv", index=False, encoding="utf-8-sig")
    hydrogen.to_csv(OUT / "hydrogen_field_audit.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# \u674e\u4fca\u8c6a\u53cc\u5806\u5b9e\u8f66\u6570\u636e\u5bf9FC\u9879\u76ee\u7684\u53ef\u7528\u6027\u5ba1\u8ba1", "",
        "## \u6838\u5fc3\u7ed3\u8bba", "",
        "1. \u4e09\u53f0\u53d1\u52a8\u673a\u90fd\u662f A/B \u53cc\u5806\u4e32\u8054\u7cfb\u7edf\uff1aA/B \u5171\u7528\u540c\u4e00\u71c3\u6599\u7535\u6c60\u7535\u6d41，\u6570\u636e\u4e2d\u6ca1\u6709\u4e24\u5806\u72ec\u7acb\u7535\u6d41\u6216\u72ec\u7acb DCDC \u6307\u4ee4\u3002\u56e0\u6b64\u53ef\u5ba1\u8ba1\u4e24\u5806\u7535\u538b/\u529f\u7387\u8d21\u732e\u548c\u4e0d\u4e00\u81f4\u9000\u5316，\u4f46\u4e0d\u662f\u5df2\u6709\u7684\u72ec\u7acb\u5806\u95f4\u53ef\u63a7\u529f\u7387\u5206\u914d\u6570\u636e\u3002",
        "2. \u201c\u76ee\u6807\u529f\u7387\u201d\u4e0e\u5b9e\u6d4b FC/DCDC \u529f\u7387\u9ad8\u5ea6\u4e00\u81f4，\u66f4\u7b26\u5408 FC \u6216 DCDC \u529f\u7387\u6307\u4ee4，\u4e0d\u7b26\u5408\u72ec\u7acb\u6574\u8f66\u9700\u6c42\u529f\u7387\u7684\u5b9a\u4e49\u3002",
        "3. Excel \u6ca1\u6709\u52a8\u529b\u7535\u6c60\u7535\u538b、\u7535\u6d41\u6216\u529f\u7387\u5b57\u6bb5，\u6240\u4ee5\u4e0d\u80fd\u628a `\u76ee\u6807\u529f\u7387-FC\u529f\u7387` \u5f53\u6210\u5b9e\u6d4b\u7535\u6c60\u529f\u7387，\u4e5f\u4e0d\u80fd\u4ece\u672c\u6570\u636e\u4e25\u683c\u590d\u539f\u539f\u8f66 FC-\u7535\u6c60\u5206\u914d\u3002",
        "4. \u8be5\u6570\u636e\u5bf9\u201c\u53cc\u5806\u9000\u5316\u4e0d\u4e00\u81f4\u6027、FC\u5de5\u4f5c\u70b9、\u6c22\u8017、\u5de5\u51b5\u66b4\u9732\u201d\u5f88\u6709\u4ef7\u503c；\u5bf9\u201c\u5b9e\u8f66\u9700\u6c42\u529f\u7387\u2192FC/\u7535\u6c60\u5206\u914d\u201d\u4e0d\u5b8c\u6574。", "",
        "## \u4e09\u65e5\u529f\u7387\u5173\u7cfb", "", metrics.to_markdown(index=False, floatfmt=".4f"), "",
        "\u4e09\u53f7、\u56db\u53f7\u7684 `\u603b\u7535\u538b/(A+B\u5e73\u5747\u8282\u7535\u538b)` \u5c3a\u5ea6\u7ea6\u4e3a 70，\u4e94\u53f7\u7ea6\u4e3a 55.5。\u56e0\u6b64\u4e0d\u5bf9\u4e09\u53f0\u8f66\u5f3a\u884c\u4f7f\u7528\u540c\u4e00\u8282\u6570；A/B \u5806\u7535\u538b\u6309\u5404\u81ea\u5e73\u5747\u7535\u538b\u5360\u6bd4\u5206\u89e3\u5b9e\u6d4b\u603b\u7535\u538b，\u5e76\u7528\u5171\u540c\u7535\u6d41\u8ba1\u7b97\u529f\u7387。", "",
        "## \u6c22\u8017\u5b57\u6bb5", "", hydrogen.to_markdown(index=False, floatfmt=".4f"), "",
        "`\u6c22\u6c14\u6d88\u8017\u7387` \u548c `\u6c22\u6c14\u5269\u4f59\u8d28\u91cf` \u5728\u6240\u9009\u6570\u636e\u4e2d\u6709\u503c，\u4f46 Excel \u8868\u5934\u672a\u5199\u5355\u4f4d，\u5f53\u524d\u53ea\u80fd\u505a\u6570\u503c\u5173\u7cfb\u548c\u76f8\u5bf9\u6c22\u8017\u5ba1\u8ba1，\u4e0d\u80fd\u672a\u7ecf\u6807\u5b9a\u5c31\u79f0\u4e3a kg/s \u6216 g/s。", "",
        "## \u5bf9FC\u4ed3\u5e93\u7814\u7a76\u8def\u7ebf\u7684\u5224\u65ad", "",
        "- **\u8f66\u8f86\u52a8\u529b\u5b66**：\u6709\u8f66\u901f、\u91cc\u7a0b、SOC，\u53ef\u6d3e\u751f\u52a0\u901f\u5ea6\u548c\u505c\u8f66\u5de5\u51b5；\u4f46\u6ca1\u6709\u6574\u8f66\u8d28\u91cf、\u5761\u5ea6、\u8e0f\u677f、\u52a8\u529b\u7535\u6c60 V/I \u548c\u7275\u5f15\u7535\u673a\u529f\u7387，\u4e0d\u80fd\u72ec\u7acb\u6821\u51c6\u771f\u5b9e\u6574\u8f66\u9700\u6c42\u529f\u7387\u3002",
        "- **\u9000\u5316**：\u6709\u957f\u65f6\u95f4\u8f74、\u5171\u540c\u7535\u6d41、A/B \u5e73\u5747/\u6700\u4f4e/70\u8282\u7535\u538b、\u6e29\u5ea6、\u538b\u529b、\u7a7a\u6c14\u6d41\u91cf、Purge，\u975e\u5e38\u9002\u5408\u7814\u7a76\u4e32\u8054\u53cc\u5806\u7684\u9000\u5316\u4e0d\u4e00\u81f4\u6027、\u5f31\u5806\u8bc6\u522b\u548c\u7535\u538b\u635f\u5931 proxy。",
        "- **\u6c22\u8017**：\u6709\u6d88\u8017\u7387\u548c\u5269\u4f59\u8d28\u91cf、\u4e5f\u6709 FC/DCDC \u529f\u7387，\u5177\u5907\u5efa\u7acb\u76f8\u5bf9\u6c22\u8017\u56fe\u8c31\u7684\u6761\u4ef6；\u7edd\u5bf9\u6c22\u8017\u9700\u5148\u786e\u8ba4\u5355\u4f4d\u548c\u4f20\u611f\u5668\u66f4\u65b0\u9891\u7387\u3002",
        "- **\u539f\u59cb\u529f\u7387\u5206\u914d**：\u53ef\u770b FC \u76ee\u6807\u2192FC\u539f\u59cb\u8f93\u51fa\u2192DCDC\u8f93\u51fa，\u53ef\u770b A/B \u4e32\u8054\u529f\u7387\u8d21\u732e；\u7f3a\u5c11\u771f\u5b9e\u6574\u8f66\u9700\u6c42\u548c\u52a8\u529b\u7535\u6c60\u529f\u7387，\u4e0d\u9002\u5408\u76f4\u63a5\u5f53\u4f5c FC-\u7535\u6c60\u539f\u8f66\u5206\u914d\u6807\u7b7e。", "",
        "## \u56fe\u8868\u8fb9\u754c", "",
        "\u6240\u6709 1 \u5929/3 \u5929\u56fe\u5747\u4f7f\u7528\u538b\u7f29\u540e\u65f6\u95f4，\u957f\u505c\u673a\u548c\u8de8\u65e5\u7a7a\u767d\u5df2\u53bb\u9664。`\u76ee\u6807-FC\u6b8b\u5dee` \u53ea\u662f\u8bed\u4e49\u5ba1\u8ba1\u91cf，\u56fe\u4e2d\u5df2\u660e\u786e\u6807\u6ce8\u201c\u4e0d\u7b49\u540c\u7535\u6c60\u529f\u7387\u201d\u3002",
    ]
    (OUT / "dual_stack_research_suitability_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
