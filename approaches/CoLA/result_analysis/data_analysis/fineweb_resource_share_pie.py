import json
import pandas as pd
import matplotlib.pyplot as plt

LANGS_JSON = "tools/two_stage_clustering/200_tier_language_groupings.json"
FW_CSV = "data_prep/base_data/fineweb2-language-distribution.csv"
RES_TSV = "data_prep/base_data/lang_resource_dataset.tsv"

MERGE_MAP = {
    "high": "High",
    "medhigh": "Mid",
    "medlow": "Low",
    "medlow*": "Low",
    "low": "Low",
    "not_enough": "Low",
}

j = json.load(open(LANGS_JSON))
langs = {l.lower() for v in j.values() for l in v.get("languages", [])}

res = pd.read_csv(RES_TSV, sep="\t")
res["lang_code"] = res["lang_code"].str.lower()
res["base"] = res["lang_code"].str.split("_").str[0]
res["tier"] = res["resource_category"].map(MERGE_MAP)
res = res.dropna(subset=["tier"]).copy()

res_map = res.set_index("lang_code")["tier"]
base_map = res.groupby("base")["tier"].agg(lambda s: s.mode().iloc[0])

tiers = {}
for code in langs:
    if code in res_map:
        tiers[code] = res_map[code]
    else:
        base = code.split("_")[0]
        if base in base_map:
            tiers[code] = base_map[base]
tiers = pd.Series(tiers)

fw = pd.read_csv(FW_CSV)
fw["base"] = fw["subset"].str.lower().str.replace("_removed", "", regex=False)
fw["documents"] = pd.to_numeric(fw["documents"], errors="coerce").fillna(0)
fw = fw[fw["base"].isin(tiers.index)]

by_lang = fw.groupby("base")["documents"].sum()
by_tier = by_lang.groupby(tiers).sum()

labels = ["High", "Mid", "Low"]
values = [float(by_tier.get(k, 0.0)) for k in labels]
colors = ["#3b82f6", "#10b981", "#f59e0b"]

total = sum(values)

legend_labels = [
    f"{label}  {((val / total) * 100.0):.2f}%" if total else f"{label}  0.00%"
    for label, val in zip(labels, values)
]

plt.figure(figsize=(4, 4))
plt.pie(
    values,
    labels=None,
    startangle=0,
    colors=colors,
    wedgeprops={"linewidth": 1, "edgecolor": "white"},
)
plt.legend(
    legend_labels,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
)
plt.tight_layout()
plt.savefig("result_analysis/data_analysis/fineweb_resource_share_pie.png", dpi=200)
