import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.decomposition import PCA
try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "fwd100_latlon.csv")
OUT_FILE  = os.path.join(BASE_DIR, "fwd_with_typology.csv")
OUT_DIR   = os.path.join(BASE_DIR, "output")
FIG_DIR   = os.path.join(OUT_DIR, "figures_revised")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {"S": "#2196F3", "M": "#FF9800", "C": "#F44336"}
TYPE_LABELS = {
    "S": "Type S – Structurally Sound",
    "M": "Type M – Moderately Deteriorated",
    "C": "Type C – Critically Deteriorated",
}
CLUSTER_FEATURES = ["D0", "SCI", "BDI", "BCI"]
RANDOM_STATE     = 42
K_FINAL          = 3
MIN_YEAR         = 2016


def savefig(name):
    fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, f"{name}_withtitle.jpg"),
                dpi=150, bbox_inches="tight", format="jpeg")

    sup = getattr(fig, "_suptitle", None)
    sup_text = sup.get_text() if sup else ""
    if sup:
        sup.set_text("")
    ax_titles = [(ax, ax.get_title()) for ax in fig.get_axes()]
    for ax, _ in ax_titles:
        ax.set_title("")

    fig.savefig(os.path.join(FIG_DIR, f"{name}_notitle.jpg"),
                dpi=150, bbox_inches="tight", format="jpeg")

    if sup:
        sup.set_text(sup_text)
    for ax, t in ax_titles:
        ax.set_title(t)

    plt.close(fig)
    print(f"  Saved {name}")


df_raw = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
df_raw.columns = [c.strip().strip('"').lstrip("﻿") for c in df_raw.columns]
for bad in [col for col in df_raw.columns if "year" in col.lower() and col != "year"]:
    df_raw.rename(columns={bad: "year"}, inplace=True)

df_raw["year"]       = df_raw["year"].astype(int)
df_raw["section_id"] = df_raw["section_id"].astype(str)
for c in (["deflection_0", "deflection_2", "deflection_5", "deflection_8",
           "minimum_life", "overlay_thickness", "temperature",
           "e1", "e2", "e3", "e4", "surface_thickness", "lat", "lon"]):
    if c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

print(f"Loaded {len(df_raw):,} rows, {df_raw['section_id'].nunique()} sections")

df_raw["D0"]  = df_raw["deflection_0"]
df_raw["SCI"] = df_raw["deflection_0"] - df_raw["deflection_2"]
df_raw["BDI"] = df_raw["deflection_2"] - df_raw["deflection_5"]
df_raw["BCI"] = df_raw["deflection_5"] - df_raw["deflection_8"]

df = df_raw.dropna(subset=CLUSTER_FEATURES).copy()
removal_log = []

for feat in CLUSTER_FEATURES:
    lo = float(np.percentile(df[feat], 2))
    hi = float(np.percentile(df[feat], 98))
    before = len(df)
    df = df[(df[feat] >= lo) & (df[feat] <= hi)].copy()
    n_rem = before - len(df)
    removal_log.append({"feature": feat, "lower_bound": round(lo, 2),
                         "upper_bound": round(hi, 2), "n_removed": n_rem,
                         "pct_removed": round(n_rem / len(df_raw) * 100, 2)})

temp_lo, temp_hi = 28.0, 55.0
before = len(df)
df = df[(df["temperature"] >= temp_lo) & (df["temperature"] <= temp_hi)].copy()
n_temp = before - len(df)
removal_log.append({"feature": "temperature", "lower_bound": temp_lo,
                     "upper_bound": temp_hi, "n_removed": n_temp,
                     "pct_removed": round(n_temp / len(df_raw) * 100, 2)})

pd.DataFrame(removal_log).to_csv(
    os.path.join(OUT_DIR, "00_outlier_removal_by_feature.csv"), index=False)
print(f"Retained {len(df):,} rows after outlier removal")

X_all = df[CLUSTER_FEATURES].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

quality_records = []
for k in range(2, 8):
    km_k = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=RANDOM_STATE)
    labels_k = km_k.fit_predict(X_scaled)
    wcss = float(km_k.inertia_)
    sil  = float(silhouette_score(X_scaled, labels_k,
                                  sample_size=min(10000, len(df)),
                                  random_state=RANDOM_STATE))
    ch   = float(calinski_harabasz_score(X_scaled, labels_k))
    db   = float(davies_bouldin_score(X_scaled, labels_k))
    quality_records.append({"k": k, "WCSS": round(wcss, 4),
                             "Silhouette": round(sil, 4),
                             "Calinski_Harabasz": round(ch, 4),
                             "Davies_Bouldin": round(db, 4)})

quality_df = pd.DataFrame(quality_records)
quality_df["WCSS_marginal_pct"] = (-quality_df["WCSS"].pct_change() * 100).round(4)
quality_df.to_csv(os.path.join(OUT_DIR, "09_cluster_quality_metrics.csv"), index=False)

km = KMeans(n_clusters=K_FINAL, n_init=50, max_iter=500, random_state=RANDOM_STATE)
df["_raw_cluster"] = km.fit_predict(X_scaled)

centroid_d0 = {
    c: df.loc[df["_raw_cluster"] == c, "D0"].mean()
    for c in range(K_FINAL)
}
rank = sorted(centroid_d0, key=centroid_d0.get)
label_map = {rank[0]: "S", rank[1]: "M", rank[2]: "C"}

df["typology"] = df["_raw_cluster"].map(label_map)
df = df.drop(columns=["_raw_cluster"])

df.to_csv(OUT_FILE, index=False)
df.to_csv(os.path.join(OUT_DIR, "01_fwd_data_with_typology.csv"), index=False)

df_c = df.copy()

centroid_vals = scaler.inverse_transform(km.cluster_centers_)
centroid_df = pd.DataFrame(centroid_vals, columns=CLUSTER_FEATURES)
centroid_df.index = [label_map[i] for i in range(K_FINAL)]
centroid_df.index.name = "typology"
centroid_df.to_csv(os.path.join(OUT_DIR, "08_kmeans_centroids.csv"))

METRICS = ["D0", "SCI", "BDI", "BCI", "e1", "e2", "e3", "e4",
           "minimum_life", "overlay_thickness", "surface_thickness", "temperature"]

full_stats = (df_c.groupby("typology")[METRICS]
              .agg(["mean", "std", "median"]).round(2))
full_stats.columns = ["_".join(c) for c in full_stats.columns]
full_stats["count"] = df_c.groupby("typology")["D0"].count()
full_stats.to_csv(os.path.join(OUT_DIR, "02_cluster_summary_statistics.csv"))
full_stats.to_csv(os.path.join(OUT_DIR, "02_full_summary_statistics.csv"))

t1_cols = ["D0", "SCI", "BDI", "BCI", "e2", "minimum_life", "overlay_thickness"]
t1_rows = []
for typ in ["S", "M", "C"]:
    sub = df_c[df_c["typology"] == typ]
    row = {"typology": typ, "n": len(sub)}
    for col in t1_cols:
        m, s = sub[col].mean(), sub[col].std()
        row[f"{col}_mean"] = round(m, 2)
        row[f"{col}_SD"]   = round(s, 2)
        row[f"{col}_mean±SD"] = f"{m:.1f} ± {s:.1f}"
    t1_rows.append(row)
pd.DataFrame(t1_rows).to_csv(
    os.path.join(OUT_DIR, "02_table1_mean_SD.csv"), index=False)

df_geo = df_c[(df_c["lat"] > 4) & (df_c["lon"] > 96)].copy()

section_df = (
    df_geo.groupby("section_id")
    .agg(
        lat            = ("lat",              "mean"),
        lon            = ("lon",              "mean"),
        n_measurements = ("D0",               "count"),
        typology       = ("typology",         lambda x: x.value_counts().index[0]),
        plurality_frac = ("typology",         lambda x: x.value_counts().iloc[0] / len(x)),
        pct_type_C     = ("typology",         lambda x: (x == "C").mean() * 100),
        mean_D0        = ("D0",               "mean"),
        mean_life      = ("minimum_life",     "mean"),
        mean_overlay   = ("overlay_thickness","mean"),
        mean_e2        = ("e2",               "mean"),
    )
    .reset_index()
    .round(2)
)
section_df.to_csv(os.path.join(OUT_DIR, "03_section_classification.csv"), index=False)

annual_counts = (
    df_c[df_c["year"] >= MIN_YEAR]
    .groupby(["year", "typology"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["S", "M", "C"], fill_value=0)
)
annual_pct = (annual_counts.div(annual_counts.sum(axis=1), axis=0) * 100).round(2)
annual_pct.columns = [f"pct_{c}" for c in annual_pct.columns]
annual = annual_counts.join(annual_pct)
annual.to_csv(os.path.join(OUT_DIR, "04_annual_typology_trend.csv"))

def maint_stats(grp):
    ov, ml = grp["overlay_thickness"], grp["minimum_life"]
    return pd.Series({
        "n":                        len(grp),
        "mean_overlay_mm":          round(ov.mean(), 2),
        "sd_overlay_mm":            round(ov.std(), 2),
        "pct_overlay_gt50mm":       round((ov > 50).mean() * 100, 2),
        "pct_overlay_gt80mm":       round((ov > 80).mean() * 100, 2),
        "mean_remaining_life_yr":   round(ml.mean(), 2),
        "sd_remaining_life_yr":     round(ml.std(), 2),
        "pct_life_lt5yr":           round((ml < 5).mean() * 100, 2),
        "pct_life_lt10yr":          round((ml < 10).mean() * 100, 2),
    })

maint = df_c.groupby("typology").apply(maint_stats)
maint.to_csv(os.path.join(OUT_DIR, "05_maintenance_requirements.csv"))

top10 = (
    df_geo[df_geo["typology"] == "C"]
    .groupby("section_id")
    .agg(n=("D0","count"), mean_D0=("D0","mean"),
         mean_life=("minimum_life","mean"),
         mean_overlay=("overlay_thickness","mean"),
         mean_e2=("e2","mean"),
         lat=("lat","mean"), lon=("lon","mean"))
    .query("n >= 10")
    .sort_values("mean_D0", ascending=False)
    .head(10)
    .reset_index()
    .round(2)
)
top10.to_csv(os.path.join(OUT_DIR, "06_top10_critical_sections.csv"), index=False)

df_c[["typology", "D0", "SCI", "BDI", "BCI", "e1", "e2", "e3", "e4"]].to_csv(
    os.path.join(OUT_DIR, "07_deflection_indices_raw.csv"), index=False)

corr_mat = df_c[["D0", "SCI", "BDI", "BCI", "temperature"]].corr().round(4)
corr_mat.to_csv(os.path.join(OUT_DIR, "10_feature_correlation_matrix.csv"))
temp_c = df_c[["D0", "SCI", "BDI", "BCI"]].corrwith(df_c["temperature"]).round(4)
pd.DataFrame({"feature": temp_c.index,
              "corr_with_temperature": temp_c.values}).to_csv(
    os.path.join(OUT_DIR, "13_temperature_deflection_correlation.csv"), index=False)

homo = (
    df_geo.groupby("section_id")
    .agg(
        typology  = ("typology", lambda x: x.value_counts().index[0]),
        plurality = ("typology", lambda x: x.value_counts().iloc[0] / len(x)),
        n         = ("typology", "count"),
    )
    .reset_index()
)

homo_summary = homo.groupby("typology").apply(lambda g: pd.Series({
    "n_sections":                int(len(g)),
    "n_homogeneous_80":          int((g["plurality"] >= 0.80).sum()),
    "pct_homogeneous_80":        round((g["plurality"] >= 0.80).mean() * 100, 1),
    "n_homogeneous_70":          int((g["plurality"] >= 0.70).sum()),
    "pct_homogeneous_70":        round((g["plurality"] >= 0.70).mean() * 100, 1),
    "mean_plurality_fraction":   round(g["plurality"].mean(), 3),
    "median_plurality_fraction": round(g["plurality"].median(), 3),
})).reset_index()
homo_summary.to_csv(os.path.join(OUT_DIR, "11_within_section_homogeneity.csv"), index=False)
homo[["section_id", "typology", "plurality", "n"]].to_csv(
    os.path.join(OUT_DIR, "11b_plurality_fraction_distribution.csv"), index=False)

sec_yr_counts = df_c.groupby("section_id")["year"].nunique()
cohort_ids = sec_yr_counts[sec_yr_counts >= 3].index.tolist()
df_cohort = df_c[(df_c["section_id"].isin(cohort_ids)) & (df_c["year"] >= MIN_YEAR)]
cohort_annual = (
    df_cohort.groupby(["year", "typology"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["S", "M", "C"], fill_value=0)
)
cohort_pct = (cohort_annual.div(cohort_annual.sum(axis=1), axis=0) * 100).round(2)
cohort_pct.columns = [f"pct_{c}" for c in cohort_pct.columns]
cohort_result = cohort_annual.join(cohort_pct)
cohort_result["n_sections"] = df_cohort.groupby("year")["section_id"].nunique()
cohort_result.to_csv(os.path.join(OUT_DIR, "12_fixed_cohort_temporal_trend.csv"))
print(f"Fixed cohort: {len(cohort_ids)} sections with ≥3 survey years")


print("Generating figures...")

ks   = quality_df["k"].values
wcss = quality_df["WCSS"].values
sil  = quality_df["Silhouette"].values
ch   = quality_df["Calinski_Harabasz"].values
db   = quality_df["Davies_Bouldin"].values
marg = quality_df["WCSS_marginal_pct"].fillna(0).values

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
ax.plot(ks, wcss, "o-", color="steelblue", lw=2, ms=8)
ax.axvline(K_FINAL, color="red", ls="--", label=f"k = {K_FINAL} (selected)")
for k_i, w_i, m_i in zip(ks[1:], wcss[1:], marg[1:]):
    ax.annotate(f"−{m_i:.1f}%", xy=(k_i, w_i), xytext=(4, 8),
                textcoords="offset points", fontsize=8, color="steelblue")
ax.set_xlabel("Number of Clusters k"); ax.set_ylabel("Within-Cluster Sum of Squares")
ax.set_title("(a) Elbow Criterion (WCSS)", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(ks, sil, "o-", color="darkgreen", lw=2, ms=8)
ax.axvline(K_FINAL, color="red", ls="--")
for k_i, s_i in zip(ks, sil):
    ax.annotate(f"{s_i:.3f}", xy=(k_i, s_i), xytext=(4, 6),
                textcoords="offset points", fontsize=8)
ax.set_xlabel("Number of Clusters k"); ax.set_ylabel("Mean Silhouette Coefficient")
ax.set_title("(b) Silhouette Coefficient", fontweight="bold"); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(ks, ch, "s-", color="darkred", lw=2, ms=8)
ax.axvline(K_FINAL, color="red", ls="--")
ax.set_xlabel("Number of Clusters k"); ax.set_ylabel("Calinski–Harabasz Index")
ax.set_title("(c) Calinski–Harabasz Index\n(higher = better separation)",
             fontweight="bold"); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(ks, db, "^-", color="purple", lw=2, ms=8)
ax.axvline(K_FINAL, color="red", ls="--")
ax.set_xlabel("Number of Clusters k"); ax.set_ylabel("Davies–Bouldin Index")
ax.set_title("(d) Davies–Bouldin Index\n(lower = better compactness)",
             fontweight="bold"); ax.grid(alpha=0.3)

fig.suptitle("Figure 1. Cluster Quality Assessment (k = 2–7)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("fig1")


rng = np.random.RandomState(RANDOM_STATE)
n_sil = min(30000, len(df_c))
idx_sil = rng.choice(len(df_c), n_sil, replace=False)
X_sil = scaler.transform(df_c.iloc[idx_sil][CLUSTER_FEATURES].astype(float))
labels_sil_raw = km.predict(X_sil)
types_sil = np.array([label_map[l] for l in labels_sil_raw])
sil_vals  = silhouette_samples(X_sil, labels_sil_raw)
mean_sil  = float(sil_vals.mean())

fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 0
for typ in ["C", "M", "S"]:
    mask  = types_sil == typ
    sv    = np.sort(sil_vals[mask])
    y_upper = y_lower + len(sv)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sv,
                     alpha=0.75, color=COLORS[typ],
                     label=f"Type {typ}  (n = {len(sv):,})")
    ax.text(-0.035, (y_lower + y_upper) / 2, f"Type {typ}",
            va="center", fontsize=8, color=COLORS[typ], fontweight="bold")
    y_lower = y_upper + 300

ax.axvline(mean_sil, color="red", ls="--", lw=2,
           label=f"Mean = {mean_sil:.3f}")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("")
ax.set_yticks([])
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
ax.set_title("Figure 2. Silhouette Plot for k = 3",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("fig2")


n_pca = min(20000, len(df_c))
idx_pca = rng.choice(len(df_c), n_pca, replace=False)
X_all_scaled = scaler.transform(df_c[CLUSTER_FEATURES].astype(float))
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca.fit(X_all_scaled)
coords = pca.transform(X_all_scaled[idx_pca])
types_pca = df_c.iloc[idx_pca]["typology"].values

fig, ax = plt.subplots(figsize=(9, 7))
for typ in ["S", "M", "C"]:
    mask = types_pca == typ
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=COLORS[typ], alpha=0.35, s=6,
               label=TYPE_LABELS[typ], rasterized=True)

scale = 2.5
for i, feat in enumerate(CLUSTER_FEATURES):
    vx, vy = pca.components_[0, i] * scale, pca.components_[1, i] * scale
    ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
    ax.text(vx * 1.18, vy * 1.18, feat, fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))

ax.axhline(0, color="gray", ls="--", lw=0.5)
ax.axvline(0, color="gray", ls="--", lw=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=10)
ax.legend(fontsize=8, markerscale=4); ax.grid(alpha=0.2)
ax.set_title("Figure 3. PCA Biplot of the Three K-means Clusters\n"
             "in Standardized Feature Space",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig3")


bowl_metrics = [
    ("D0",  "D0 – Peak Deflection (µm)"),
    ("SCI", "SCI – Surface Curvature Index (µm)"),
    ("BDI", "BDI – Base Damage Index (µm)"),
    ("BCI", "BCI – Base Curvature Index (µm)"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (col, title) in zip(axes.flat, bowl_metrics):
    data = [df_c[df_c["typology"] == t][col].dropna() for t in ["S", "M", "C"]]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", ms=2, alpha=0.3))
    for patch, typ in zip(bp["boxes"], ["S", "M", "C"]):
        patch.set_facecolor(COLORS[typ])
        patch.set_alpha(0.75)
    ax.set_xticklabels(["Type S", "Type M", "Type C"], fontsize=9)
    ax.set_ylabel(title, fontsize=9)
    ax.set_title(col, fontsize=10, fontweight="bold"); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 4. Deflection Bowl Indices by Pavement Structural Typology",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("fig4")


counts     = df_c["typology"].value_counts().reindex(["S", "M", "C"])
sec_counts = section_df["typology"].value_counts().reindex(["S", "M", "C"])
labels_pie = [TYPE_LABELS[t] for t in ["S", "M", "C"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
wedges, texts, autotexts = ax1.pie(
    counts, labels=labels_pie,
    colors=[COLORS["S"], COLORS["M"], COLORS["C"]],
    autopct="%1.1f%%", startangle=140,
    textprops={"fontsize": 9}, pctdistance=0.75)
for at in autotexts:
    at.set_fontweight("bold")
ax1.set_title("(a) Proportion of FWD Measurements", fontsize=10, fontweight="bold")

bars = ax2.bar(["Type S", "Type M", "Type C"], sec_counts,
               color=[COLORS["S"], COLORS["M"], COLORS["C"]],
               edgecolor="black", lw=0.5)
for bar, val in zip(bars, sec_counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(val), ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("Number of Highway Sections", fontsize=10)
ax2.set_title("(b) Sections by Dominant Typology", fontsize=10, fontweight="bold")
ax2.set_ylim(0, sec_counts.max() * 1.15); ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 5. Proportion of FWD Measurements and Sections by Dominant Typology",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig5")


df_mod = df_c.copy()
for col in ["e1", "e2", "e3", "e4"]:
    lo, hi = np.percentile(df_mod[col].dropna(), [2, 98])
    df_mod = df_mod[(df_mod[col] >= lo) & (df_mod[col] <= hi)]

moduli_cols = [("e1", "AC Layer (e₁)"), ("e2", "Base (e₂)"),
               ("e3", "Subbase (e₃)"), ("e4", "Subgrade (e₄)")]
fig, axes = plt.subplots(1, 4, figsize=(13, 5))
for ax, (col, title) in zip(axes, moduli_cols):
    means = [df_mod[df_mod["typology"] == t][col].mean() for t in ["S", "M", "C"]]
    sds   = [df_mod[df_mod["typology"] == t][col].std()  for t in ["S", "M", "C"]]
    bars  = ax.bar(["S", "M", "C"], means, yerr=sds, capsize=5,
                   color=[COLORS["S"], COLORS["M"], COLORS["C"]],
                   edgecolor="black", lw=0.5, alpha=0.85,
                   error_kw=dict(lw=1))
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Typology", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("Elastic Modulus (MPa)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(val + sd, ax.get_ylim()[1]) * 1.02,
                f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")

fig.suptitle("Figure 6. Mean Layer Elastic Moduli by Pavement Structural Typology (±1 SD)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig6")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bins = np.arange(0.30, 1.05, 0.05)
for typ in ["S", "M", "C"]:
    sub = homo[homo["typology"] == typ]["plurality"]
    ax1.hist(sub, bins=bins, alpha=0.65, color=COLORS[typ],
             label=f"Type {typ} (n = {len(sub)})")
ax1.axvline(0.80, color="black", ls="--", lw=2, label="80% homogeneity threshold")
ax1.set_xlabel("Plurality Fraction (dominant typology share per section)", fontsize=9)
ax1.set_ylabel("Number of Sections", fontsize=9)
ax1.set_title("(a) Distribution of Section-Level\nTypology Homogeneity", fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

x_pos = np.arange(3)
labels_bar = ["Type S", "Type M", "Type C"]
for i, typ in enumerate(["S", "M", "C"]):
    row = homo_summary[homo_summary["typology"] == typ].iloc[0]
    n_homo  = int(row["n_homogeneous_80"])
    n_mixed = int(row["n_sections"]) - n_homo
    pct     = row["pct_homogeneous_80"]
    ax2.bar(i, n_homo,  color=COLORS[typ], edgecolor="black", lw=0.5)
    ax2.bar(i, n_mixed, bottom=n_homo,
            color=COLORS[typ], edgecolor="black", lw=0.5, hatch="///", alpha=0.55)
    ax2.text(i, int(row["n_sections"]) + 3, f"{pct:.0f}% homogeneous",
             ha="center", fontsize=9, fontweight="bold")

ax2.set_xticks(x_pos); ax2.set_xticklabels(labels_bar)
ax2.set_ylabel("Number of Sections", fontsize=9)
ax2.set_title("(b) Homogeneous (solid) vs Mixed (hatched) Sections\n"
              "per Typology (≥80% threshold)", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 7. Within-Section Typology Homogeneity: "
             "Plurality Fraction and ≥80% Threshold",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig7")


section_df["x"] = section_df["lon"] * 20037508.34 / 180
section_df["y"] = (
    np.log(np.tan((90 + section_df["lat"]) * np.pi / 360)) / (np.pi / 180)
    * 20037508.34 / 180
)

fig, ax = plt.subplots(figsize=(7, 10))
for typ in ["S", "M", "C"]:
    g = section_df[section_df["typology"] == typ]
    ax.scatter(g["x"], g["y"], c=COLORS[typ], s=25, alpha=0.85,
               edgecolors="white", lw=0.3,
               label=f"Type {typ}  (n = {len(g)})", zorder=4)
ax.set_xlim(10_800_000, 11_750_000)
ax.set_ylim(600_000, 2_350_000)
if HAS_CTX:
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=7, attribution_size=6)
    except Exception as e:
        print(f"basemap unavailable: {e}")
ax.set_axis_off()
ax.legend(title="Structural Typology", fontsize=9, title_fontsize=9,
          loc="lower left", framealpha=0.85)
ax.set_title("Figure 8. Spatial Distribution of Dominant Typology\n"
             "Thai National Highways (2014–2023)",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
savefig("fig8")


fig, ax = plt.subplots(figsize=(9, 5))
for typ in ["S", "M", "C"]:
    col = f"pct_{typ}"
    if col not in annual_pct.columns:
        continue
    ax.plot(annual_pct.index, annual_pct[col], "o-",
            color=COLORS[typ], lw=2, ms=7, label=f"Type {typ}")
    ax.annotate(f"{annual_pct[col].iloc[-1]:.1f}%",
                xy=(annual_pct.index[-1], annual_pct[col].iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=COLORS[typ])
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Percentage of FWD Measurements (%)", fontsize=11)
ax.set_xticks(annual_pct.index)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 70)
ax.set_title("Figure 9. Annual Proportion of FWD Measurements\n"
             "by Pavement Structural Typology (2016–2023)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig9")


common_years = annual_pct.index.intersection(cohort_result.index)

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
for ax, typ in zip(axes, ["S", "M", "C"]):
    col = f"pct_{typ}"
    ax.plot(annual_pct.index, annual_pct[col], "o-",
            color=COLORS[typ], lw=2, ms=7, label="Full network")
    if col in cohort_result.columns:
        ax.plot(common_years, cohort_result.loc[common_years, col], "s--",
                color=COLORS[typ], lw=2, ms=7, alpha=0.65,
                label=f"Fixed cohort\n(n={len(cohort_ids)} sections)")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(f"Type {typ} Proportion (%)", fontsize=10)
    ax.set_title(f"Type {typ} – {TYPE_LABELS[typ].split('–')[1].strip()}",
                 fontweight="bold", fontsize=9)
    ax.set_xticks(annual_pct.index)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("Figure 10. Annual Structural Typology Trends: "
             "Full Network vs. Fixed Cohort",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig10")


dfC_ov = df_c[df_c["typology"] == "C"]["overlay_thickness"].dropna()
dfC_ov = dfC_ov[dfC_ov > 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
ax1.hist(dfC_ov, bins=30, color=COLORS["C"], edgecolor="black", lw=0.4, alpha=0.82)
ax1.axvline(dfC_ov.median(), color="black", lw=2, ls="--",
            label=f"Median = {dfC_ov.median():.1f} mm")
ax1.axvline(50, color="orange", lw=2, ls=":", label="50 mm threshold")
ax1.set_xlabel("Required Overlay Thickness (mm)", fontsize=10)
ax1.set_ylabel("Count of FWD Measurements", fontsize=10)
ax1.set_title("(a) Distribution of Overlay Requirements\n(Type C — overlay > 0 mm only)",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

bins_ov   = [0, 10, 20, 30, 50, 75, 100, float(dfC_ov.max()) + 1]
labels_ov = ["<10", "10–20", "20–30", "30–50", "50–75", "75–100", ">100"]
cuts_ov   = pd.cut(dfC_ov, bins=bins_ov, labels=labels_ov)
counts_ov = cuts_ov.value_counts().reindex(labels_ov)
pct_ov    = counts_ov / len(dfC_ov) * 100

bars = ax2.barh(labels_ov, pct_ov, color=COLORS["C"],
                edgecolor="black", lw=0.4, alpha=0.82)
for bar, val in zip(bars, pct_ov):
    ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=8)
ax2.set_xlabel("Percentage of Type C Measurements (%)", fontsize=10)
ax2.set_ylabel("Overlay Thickness Band (mm)", fontsize=10)
ax2.set_title("(b) Breakdown by Overlay Thickness Band\n(Type C Sections Only)",
              fontsize=10, fontweight="bold")
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("Figure 11. Required Overlay Thickness Distribution for "
             "Type C (Critically Deteriorated) Sections",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig11")

print("Done.")
