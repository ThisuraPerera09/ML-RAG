"""
generate_charts.py
------------------
Generates exploratory charts from raw_houses.csv and saves them to plots/.
Run with:  python generate_charts.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_PATH  = "data/raw_houses.csv"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).dropna(subset=["price_lkr"])
df = df[df["price_lkr"] > 0]

PURPLE = "#6366f1"
BG     = "#1a1d27"
GRID   = "#2a2d3e"
TEXT   = "#e2e8f0"

def apply_dark(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.7)

def lkr_fmt(x, _):
    if x >= 1_000_000: return f"LKR {x/1_000_000:.1f}M"
    return f"LKR {x/1_000:.0f}K"


# ── 1. Price distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["price_lkr"], bins=60, color=PURPLE, edgecolor=BG, linewidth=0.4)
ax.set_title("House Price Distribution")
ax.set_xlabel("Price (LKR)", color=TEXT)
ax.set_ylabel("Number of Properties", color=TEXT)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_price_distribution.png", dpi=150)
plt.close()
print("Saved 01_price_distribution.png")


# ── 2. Avg price by district ──────────────────────────────────────────────────
avg_dist = df.groupby("district")["price_lkr"].mean().sort_values(ascending=False)
colors   = [PURPLE if i == 0 else "#4f46e5" for i in range(len(avg_dist))]
fig, ax  = plt.subplots(figsize=(9, 4))
ax.bar(avg_dist.index, avg_dist.values, color=colors, edgecolor=BG)
ax.set_title("Average House Price by District")
ax.set_ylabel("Avg Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_avg_price_by_district.png", dpi=150)
plt.close()
print("Saved 02_avg_price_by_district.png")


# ── 3. Avg price by property type ─────────────────────────────────────────────
avg_type = df.groupby("property_type")["price_lkr"].mean().sort_values(ascending=False)
fig, ax  = plt.subplots(figsize=(7, 4))
ax.bar(avg_type.index, avg_type.values, color=PURPLE, edgecolor=BG)
ax.set_title("Average Price by Property Type")
ax.set_ylabel("Avg Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_avg_price_by_property_type.png", dpi=150)
plt.close()
print("Saved 03_avg_price_by_property_type.png")


# ── 4. Bedrooms vs price (box plot) ──────────────────────────────────────────
bed_groups = [df[df["bedrooms"] == b]["price_lkr"].values for b in sorted(df["bedrooms"].unique())]
bed_labels = [f"{b} bed" for b in sorted(df["bedrooms"].unique())]
fig, ax    = plt.subplots(figsize=(9, 4))
bp = ax.boxplot(bed_groups, tick_labels=bed_labels, patch_artist=True,
                medianprops=dict(color="#f59e0b", linewidth=2),
                whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
                flierprops=dict(marker="o", color=PURPLE, alpha=0.3, markersize=3))
for patch in bp["boxes"]:
    patch.set_facecolor(PURPLE)
    patch.set_alpha(0.7)
ax.set_title("Price Distribution by Bedroom Count")
ax.set_ylabel("Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_price_by_bedrooms.png", dpi=150)
plt.close()
print("Saved 04_price_by_bedrooms.png")


# ── 5. Land size vs price scatter ─────────────────────────────────────────────
sample = df.dropna(subset=["land_perches", "floor_area_sqft"]).sample(min(1000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(sample["land_perches"], sample["price_lkr"], alpha=0.3, s=8, color=PURPLE)
m, b = np.polyfit(sample["land_perches"], sample["price_lkr"], 1)
x_line = np.linspace(sample["land_perches"].min(), sample["land_perches"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#f59e0b", linewidth=1.5, label="Trend")
ax.set_title("Land Size vs House Price")
ax.set_xlabel("Land Size (perches)", color=TEXT)
ax.set_ylabel("Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
ax.legend(facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/05_land_vs_price.png", dpi=150)
plt.close()
print("Saved 05_land_vs_price.png")


# ── 6. Floor area vs price scatter ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(sample["floor_area_sqft"], sample["price_lkr"], alpha=0.3, s=8, color=PURPLE)
m, b = np.polyfit(sample["floor_area_sqft"], sample["price_lkr"], 1)
x_line = np.linspace(sample["floor_area_sqft"].min(), sample["floor_area_sqft"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#f59e0b", linewidth=1.5, label="Trend")
ax.set_title("Floor Area vs House Price")
ax.set_xlabel("Floor Area (sqft)", color=TEXT)
ax.set_ylabel("Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
ax.legend(facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/06_sqft_vs_price.png", dpi=150)
plt.close()
print("Saved 06_sqft_vs_price.png")


# ── 7. Premium features comparison ───────────────────────────────────────────
features   = ["has_pool", "has_garage", "furnished"]
labels     = ["Pool", "Garage", "Furnished"]
with_avg   = [df[df[f] == 1]["price_lkr"].mean() for f in features]
without_avg = [df[df[f] == 0]["price_lkr"].mean() for f in features]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - 0.2, with_avg,    0.35, label="With",    color=PURPLE,   edgecolor=BG)
ax.bar(x + 0.2, without_avg, 0.35, label="Without", color="#4f46e5", edgecolor=BG)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Impact of Premium Features on Price")
ax.set_ylabel("Avg Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
ax.legend(facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/07_premium_features.png", dpi=150)
plt.close()
print("Saved 07_premium_features.png")


# ── 8. Property type distribution (pie) ───────────────────────────────────────
counts = df["property_type"].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
wedges, texts, autotexts = ax.pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=["#6366f1", "#4f46e5", "#818cf8", "#a5b4fc"],
    startangle=140,
    wedgeprops=dict(edgecolor=BG, linewidth=1.5),
)
for t in texts:     t.set_color(TEXT)
for t in autotexts: t.set_color(BG); t.set_fontsize(9)
ax.set_title("Property Type Distribution", color=TEXT)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/08_property_type_distribution.png", dpi=150)
plt.close()
print("Saved 08_property_type_distribution.png")


# ── 9. Age vs price ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(sample["age_years"], sample["price_lkr"], alpha=0.3, s=8, color=PURPLE)
m, b = np.polyfit(sample["age_years"], sample["price_lkr"], 1)
x_line = np.linspace(0, sample["age_years"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#f59e0b", linewidth=1.5, label="Trend")
ax.set_title("Property Age vs Price")
ax.set_xlabel("Age (years)", color=TEXT)
ax.set_ylabel("Price (LKR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
ax.legend(facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
apply_dark(fig, ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lkr_fmt))
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/09_age_vs_price.png", dpi=150)
plt.close()
print("Saved 09_age_vs_price.png")


print(f"\nAll charts saved to '{OUTPUT_DIR}/'")
