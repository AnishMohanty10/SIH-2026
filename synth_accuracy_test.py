"""
MPLADS Anomaly Detection — Synthetic Test Set + Accuracy Measurement
=====================================================================
Confirmed thresholds (read from actual data/notebook before running):
  Speed outlier : days_to_sanction < 12  OR  > 300  (5th/95th pct, combined)
  Cost outlier  : |cost_zscore| > 3  (per work_category)
  Stalled       : Work Status != 'Work Completed' AND days_since_sanction > 365
  ML anomaly    : Isolation Forest contamination=0.05 on 3 features
  App threshold : Total_Risk_Score >= 1 → "any signal fired"
                  Total_Risk_Score >= 3 → priority queue
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

TODAY = date(2026, 9, 5)
REPO  = "/Users/asd/Desktop/SIH-2026"
RAW_CSV    = f"{REPO}/Works Sanctioned.csv"
OUTPUT_CSV = f"{REPO}/sih_audited_works_final_TEST.csv"   # never overwrites production

# ── Confirmed per-category stats (from real data) ─────────────────────────────
CAT_STATS = {
    "Other":                    dict(mean=521759,  std=976214),
    "Road/Infrastructure":      dict(mean=590140,  std=621973),
    "Lighting":                 dict(mean=344180,  std=1073031),
    "Community Building":       dict(mean=652255,  std=758066),
    "Education":                dict(mean=689241,  std=825642),
    "Water Access":             dict(mean=285633,  std=478414),
    "Religious Sites":          dict(mean=474344,  std=502287),
    "Paving/Flooring":         dict(mean=594495,  std=499933),
    "Sports/Recreation":        dict(mean=649051,  std=956200),
    "Electrification":          dict(mean=352367,  std=587425),
    "Boundary Wall":            dict(mean=615223,  std=590298),
    "Drainage":                 dict(mean=692058,  std=601455),
    "PDS/Ration Infrastructure":dict(mean=991250,  std=1214039),
    "Cremation/Burial Grounds": dict(mean=525235,  std=666156),
    "Sanitation/Toilets":      dict(mean=665134,  std=635665),
    "CCTV/Security":            dict(mean=742936,  std=1737695),
    "Anganwadi/Childcare":      dict(mean=1274929, std=658889),
}

# Keywords that map a description to a category (mirrors notebook categorize_work)
CAT_KEYWORDS = {
    "Road/Infrastructure":      "road",
    "Community Building":       "bhavan",
    "Education":                "school",
    "Water Access":             "borewell",
    "Lighting":                 "solar light pole",
    "Drainage":                 "cc drain drainage",
    "CCTV/Security":            "cctv camera",
    "Sanitation/Toilets":       "toilet sochalay",
    "Sports/Recreation":        "playground ground",
    "Boundary Wall":            "compound wall boundary",
    "Anganwadi/Childcare":      "anganwadi",
}
# "Other" = no matching keyword → use generic description

def cat_desc(cat):
    """Return a work description fragment that will map to the requested category."""
    m = {
        "Road/Infrastructure":       "Construction of road and culvert",
        "Community Building":        "Construction of Community Bhavan",
        "Education":                 "Construction of school building",
        "Water Access":              "Installation of borewell and hand pump",
        "Lighting":                  "Installation of solar light pole",
        "Drainage":                  "Construction of cc drain and drainage channel",
        "CCTV/Security":             "Installation of CCTV camera surveillance system",
        "Sanitation/Toilets":        "Construction of toilet and sochalay block",
        "Sports/Recreation":         "Development of playground and ground levelling",
        "Boundary Wall":             "Construction of compound wall and boundary fencing",
        "Anganwadi/Childcare":       "Construction of anganwadi centre",
        "Other":                     "Renovation of general infrastructure facility",
    }
    return m.get(cat, "General civil work")

def fmt_date(d):
    return d.strftime("%d-%b-%Y")   # 05-Sep-2026

def amount_for(cat, z=0.0):
    """Return a Sanction Amount (int) corresponding to cat_mean + z*cat_std."""
    s = CAT_STATS.get(cat, CAT_STATS["Other"])
    val = int(s["mean"] + z * s["std"])
    return max(10000, val)   # floor at 10k to stay realistic

def make_row(idx, cat, rec_date, sanc_date, amount, status,
             expected_flag, signals, note=""):
    """Build one synthetic CSV row dict."""
    mp_id = f"TEST_MP_{chr(65 + (idx % 26))}{idx // 26 + 1:02d}"
    return {
        "Sr. No.":                   f"T{idx:04d}",
        "Work category":             "Normal/Others",   # match real CSV dominant category
        "Work":                      f"WS/TEST/2025-2026/SYNTH{idx:04d}-{cat_desc(cat)[:40]}",
        "State":                     "TEST_STATE_A" if idx % 2 == 0 else "TEST_STATE_B",
        "IDA":                       f"TEST_IDA_DISTRICT{idx % 5 + 1}",
        "Hon'ble Members of Parliament": mp_id,
        "Constituency":              f"TEST_CONSTITUENCY_{idx % 10 + 1:02d}",
        "Work description":          f"TEST_WORK: {cat_desc(cat)} at Test Village Pry No SYNTH{idx:04d}",
        "Recommended date":          fmt_date(rec_date),
        "Sanction Date":             fmt_date(sanc_date),
        "Sanction Amount ( ₹ )":     str(amount),
        "Work Status":               status,
        # Extra columns (not in original schema — pipeline ignores them)
        "is_synthetic_test_row":     True,
        "expected_should_be_flagged":expected_flag,
        "expected_signals":          signals,
    }

rows = []
idx = 1

# ══════════════════════════════════════════════════════════════════════════════
# CLEAN ROWS (120 total)
# ══════════════════════════════════════════════════════════════════════════════

# 50 regular completed works — normal speed, normal cost, completed
for i in range(50):
    cat = ["Road/Infrastructure","Community Building","Education","Water Access","Lighting"][i % 5]
    days_speed = 60 + (i % 12) * 18   # 60–258 days → well inside 12–300
    sanc = TODAY - timedelta(days=200 + i % 60)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=round((i % 7 - 3) * 0.4, 2))   # z in [-1.2, +1.2]
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         False, "none", "clean_completed"))
    idx += 1

# 40 regular in-progress works — recently sanctioned (not yet stale)
for i in range(40):
    cat = ["Other","Drainage","Sanitation/Toilets","Boundary Wall","Sports/Recreation"][i % 5]
    days_speed = 50 + (i % 8) * 20    # 50–190
    sanc = TODAY - timedelta(days=100 + i % 80)   # 100–179 days ago → not stalled
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=round((i % 5 - 2) * 0.5, 2))   # z in [-1.0, +1.0]
    status = ["Physical Inspection","Vendor Identification","Sanction"][i % 3]
    rows.append(make_row(idx, cat, rec, sanc, amt, status,
                         False, "none", "clean_inprogress"))
    idx += 1

# 15 borderline speed — exactly AT the threshold boundary (not beyond it)
# days_to_sanction = 12 (== lower) → NOT flagged; days_to_sanction = 300 (== upper) → NOT flagged
for i in range(15):
    cat = "Community Building"
    if i < 8:
        days_speed = 12    # exactly at lower boundary → condition is < 12, so NOT flagged
    else:
        days_speed = 300   # exactly at upper boundary → condition is > 300, so NOT flagged
    sanc = TODAY - timedelta(days=180 + i * 5)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=0.5)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         False, "none", "borderline_speed_inside"))
    idx += 1

# 15 borderline cost — z-score ~2.8 (just below |z|>3 threshold) → NOT flagged
for i in range(15):
    cat = ["Education","Road/Infrastructure","Other"][i % 3]
    days_speed = 90 + i * 10
    sanc = TODAY - timedelta(days=200 + i * 3)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=2.8)    # just below 3.0 → NOT flagged
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         False, "none", "borderline_cost_inside"))
    idx += 1

print(f"Clean rows generated: {len(rows)} (target 120)")
clean_count = len(rows)

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALOUS ROWS (130 total)
# ══════════════════════════════════════════════════════════════════════════════

# ── Signal 1a: Speed outlier FAST (days_to_sanction ≤ 5, far below threshold of 12) ──
# 20 rows
for i in range(20):
    cat = ["Community Building","Education","Road/Infrastructure","Other","Water Access"][i % 5]
    days_speed = i % 5       # 0–4 days → clearly below 12
    sanc = TODAY - timedelta(days=300 + i * 5)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=round((i % 3 - 1) * 0.6, 2))   # normal cost
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "speed_outlier_fast"))
    idx += 1

# ── Signal 1b: Speed outlier SLOW (days_to_sanction ≥ 350, above threshold of 300) ──
# 15 rows
for i in range(15):
    cat = ["Drainage","Boundary Wall","Lighting"][i % 3]
    days_speed = 350 + i * 15    # 350–560 → clearly above 300
    sanc = TODAY - timedelta(days=400 + i * 10)
    rec  = sanc - timedelta(days=days_speed)
    if rec > sanc:  # safety
        rec = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=0.3)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Physical Inspection",
                         True, "speed_outlier_slow"))
    idx += 1

# ── Signal 1c: Borderline speed JUST OUTSIDE threshold ──
# 5 rows (just past boundary → should be flagged)
for i in range(5):
    cat = "Road/Infrastructure"
    days_speed = 11 if i < 3 else 301    # 11 < 12 → flagged; 301 > 300 → flagged
    sanc = TODAY - timedelta(days=250 + i * 20)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=0.4)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "speed_outlier_borderline"))
    idx += 1

# ── Signal 2a: Cost outlier HIGH (z > 3) ──
# 20 rows across 5 different categories
for i in range(20):
    cat = ["Education","Community Building","Road/Infrastructure","Drainage","Water Access"][i % 5]
    z = 3.5 + (i % 4) * 0.5    # z in [3.5, 5.0]
    days_speed = 80 + i * 7
    sanc = TODAY - timedelta(days=200 + i * 4)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "cost_outlier_high"))
    idx += 1

# ── Signal 2b: Cost outlier LOW (z < -3) ──
# 5 rows — very cheap projects for high-cost categories
for i in range(5):
    cat = ["Anganwadi/Childcare","PDS/Ration Infrastructure","Education"][i % 3]
    z = -3.5 - i * 0.3    # z in [-3.5, -4.7]
    days_speed = 70 + i * 10
    sanc = TODAY - timedelta(days=180 + i * 10)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "cost_outlier_low"))
    idx += 1

# ── Signal 2c: Cost borderline JUST OUTSIDE (z ~3.1) ──
# 5 rows
for i in range(5):
    cat = ["Sports/Recreation","CCTV/Security","Boundary Wall"][i % 3]
    z = 3.1 + i * 0.05    # just above 3.0
    days_speed = 100 + i * 15
    sanc = TODAY - timedelta(days=200 + i * 8)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "cost_outlier_borderline"))
    idx += 1

# ── Signal 3: Stalled ONLY (>365 days since sanction, not completed) ──
# 20 rows — all sanctioned well before 2025-09-05 with non-completed status
for i in range(20):
    cat = ["Other","Physical Inspection","Vendor Identification"][i % 3]  # status, not cat
    work_cat = ["Community Building","Road/Infrastructure","Other","Lighting"][i % 4]
    days_since = 400 + i * 10    # 400–590 days since sanction → stalled
    sanc = TODAY - timedelta(days=days_since)
    days_speed = 60 + i * 5
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(work_cat, z=round((i % 3 - 1) * 0.5, 2))
    status = ["Physical Inspection","Vendor Identification","Sanction","Work partially Completed"][i % 4]
    rows.append(make_row(idx, work_cat, rec, sanc, amt, status,
                         True, "stalled"))
    idx += 1

# ── Combination: Speed FAST + Cost HIGH ──
# 8 rows
for i in range(8):
    cat = ["Education","Community Building"][i % 2]
    days_speed = i % 4     # 0–3 → speed outlier fast
    z = 3.8 + i * 0.2     # high cost outlier
    sanc = TODAY - timedelta(days=300 + i * 20)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "speed_outlier_fast,cost_outlier_high"))
    idx += 1

# ── Combination: Speed SLOW + Cost HIGH ──
# 5 rows
for i in range(5):
    cat = "Road/Infrastructure"
    days_speed = 400 + i * 20    # slow
    z = 4.0 + i * 0.3            # high cost
    sanc = TODAY - timedelta(days=500 + i * 10)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Work Completed",
                         True, "speed_outlier_slow,cost_outlier_high"))
    idx += 1

# ── Combination: Speed FAST + Stalled ──
# 7 rows — suspiciously fast approval, then never completed
for i in range(7):
    cat = ["Community Building","Drainage"][i % 2]
    days_speed = i % 4      # 0–3 days
    days_since = 420 + i * 15
    sanc = TODAY - timedelta(days=days_since)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=0.4)
    status = ["Physical Inspection","Sanction","Vendor Identification"][i % 3]
    rows.append(make_row(idx, cat, rec, sanc, amt, status,
                         True, "speed_outlier_fast,stalled"))
    idx += 1

# ── Combination: Cost HIGH + Stalled ──
# 10 rows
for i in range(10):
    cat = ["Education","Water Access","Boundary Wall"][i % 3]
    z = 3.5 + i * 0.2
    days_since = 400 + i * 12
    sanc = TODAY - timedelta(days=days_since)
    days_speed = 80 + i * 8
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    status = ["Physical Inspection","Vendor Identification"][i % 2]
    rows.append(make_row(idx, cat, rec, sanc, amt, status,
                         True, "cost_outlier_high,stalled"))
    idx += 1

# ── Triple: Speed FAST + Cost HIGH + Stalled ──
# 5 rows — designed to score 3 on the rule-based signals
for i in range(5):
    cat = ["Anganwadi/Childcare","PDS/Ration Infrastructure"][i % 2]
    days_speed = i % 3          # 0–2 days fast
    z = 4.0 + i * 0.5           # high cost
    days_since = 450 + i * 20
    sanc = TODAY - timedelta(days=days_since)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Sanction",
                         True, "speed_outlier_fast,cost_outlier_high,stalled"))
    idx += 1

# ── Designed for ML anomaly (unusual multi-variable combinations) ──
# 5 rows — mid-range individually but unusual in combination
for i in range(5):
    cat = "Other"
    days_speed = [250, 260, 270, 255, 265][i]   # near 95th pct but not over it
    days_since = [360, 362, 358, 355, 363][i]   # near stalled but not over it
    z_val = [2.7, 2.9, 2.8, 2.75, 2.85][i]     # near cost threshold but not over it
    sanc = TODAY - timedelta(days=days_since)
    rec  = sanc - timedelta(days=days_speed)
    amt  = amount_for(cat, z=z_val)
    rows.append(make_row(idx, cat, rec, sanc, amt, "Physical Inspection",
                         True, "ml_anomaly_designed"))
    idx += 1

anomalous_count = len(rows) - clean_count
print(f"Anomalous rows generated: {anomalous_count} (target 130)")
print(f"Total synthetic rows: {len(rows)}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2A — APPEND TO Works Sanctioned.csv
# ══════════════════════════════════════════════════════════════════════════════
synth_df = pd.DataFrame(rows)

# Load original, append synthetic rows (extra columns will be NaN for real rows — that's fine)
orig = pd.read_csv(RAW_CSV, low_memory=False)
print(f"\nOriginal rows: {len(orig)}")
combined_raw = pd.concat([orig, synth_df], ignore_index=True)
combined_raw.to_csv(RAW_CSV, index=False)
print(f"Combined rows written to Works Sanctioned.csv: {len(combined_raw)}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2B — FULL PIPELINE RERUN (exact notebook logic)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Running full pipeline ===")
works = pd.read_csv(RAW_CSV, low_memory=False)
works.columns = works.columns.str.strip()

# --- Date parsing ---
works['Recommended date'] = pd.to_datetime(works['Recommended date'], format='%d-%b-%Y', errors='coerce')
works['Sanction Date']    = pd.to_datetime(works['Sanction Date'],    format='%d-%b-%Y', errors='coerce')
works['Sanction Amount ( ₹ )'] = pd.to_numeric(works['Sanction Amount ( ₹ )'], errors='coerce')
works = works.rename(columns={'Sanction Amount ( ₹ )': 'Sanction_Amount'})
works_clean = works.dropna(subset=['Recommended date', 'Sanction Date', 'Sanction_Amount']).copy()
print(f"Rows after cleaning: {len(works_clean)}")

# --- Signal 1: Speed outlier ---
works_clean['days_to_sanction'] = (works_clean['Sanction Date'] - works_clean['Recommended date']).dt.days
lower_threshold = works_clean['days_to_sanction'].quantile(0.05)
upper_threshold = works_clean['days_to_sanction'].quantile(0.95)
print(f"Speed thresholds: lower={lower_threshold}, upper={upper_threshold}")
works_clean['is_speed_outlier'] = (
    (works_clean['days_to_sanction'] < lower_threshold) |
    (works_clean['days_to_sanction'] > upper_threshold)
)

# --- work_category (notebook cell 8 keyword logic) ---
def categorize_work(description):
    d = str(description).lower()
    if any(k in d for k in ['road','paver','pcc','bridge','febar block']): return 'Road/Infrastructure'
    elif any(k in d for k in ['bhavan','community hall','community center','community centre','shed']): return 'Community Building'
    elif any(k in d for k in ['school','college','vidyalay','classroom','laborator']): return 'Education'
    elif any(k in d for k in ['drainage','cc drain']): return 'Drainage'
    elif any(k in d for k in ['hand pump','handpump','tubewell','borewell','ro plant','water']): return 'Water Access'
    elif any(k in d for k in ['cctv','camera']): return 'CCTV/Security'
    elif any(k in d for k in ['solar','light','pole']): return 'Lighting'
    elif any(k in d for k in ['compound wall','boundary']): return 'Boundary Wall'
    elif any(k in d for k in ['toilet','sochalay','shauchalay']): return 'Sanitation/Toilets'
    elif any(k in d for k in ['ground','playground','stadium']): return 'Sports/Recreation'
    elif any(k in d for k in ['anganwadi']): return 'Anganwadi/Childcare'
    else: return 'Other'

works_clean['work_category'] = works_clean['Work description'].apply(categorize_work)

# --- Signal 2: Cost z-score per category ---
cat_stats = works_clean.groupby('work_category')['Sanction_Amount'].agg(['mean','std','count']).reset_index()
cat_stats.columns = ['work_category','category_mean','category_std','category_count']
works_clean = works_clean.merge(cat_stats, on='work_category', how='left')
works_clean['cost_zscore'] = (works_clean['Sanction_Amount'] - works_clean['category_mean']) / works_clean['category_std']
works_clean['is_cost_outlier'] = works_clean['cost_zscore'].abs() > 3

# --- Signal 3: Stalled ---
ref_date = pd.Timestamp(TODAY)
works_clean['days_since_sanction'] = (ref_date - works_clean['Sanction Date']).dt.days
works_clean['is_stalled'] = (
    (works_clean['Work Status'] != 'Work Completed') &
    (works_clean['days_since_sanction'] > 365)
)

# --- Duplicate flag (TF-IDF per MP) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
works_clean['is_duplicate_flag'] = False
for mp, group in works_clean.groupby("Hon'ble Members of Parliament"):
    if len(group) < 2:
        continue
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(group['Work description'].fillna(''))
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)
        has_duplicate = (sim_matrix > 0.85).any(axis=1)
        duplicate_indices = group.index[has_duplicate]
        works_clean.loc[duplicate_indices, 'is_duplicate_flag'] = True
    except ValueError:
        continue
print(f"Duplicate flag: {works_clean['is_duplicate_flag'].sum()} rows")

# --- Signal 4: Isolation Forest ---
from sklearn.ensemble import IsolationForest
ml_features = works_clean[['days_to_sanction','cost_zscore','days_since_sanction']].copy()
ml_features['cost_zscore'] = ml_features['cost_zscore'].abs().fillna(0)
ml_features['days_to_sanction'] = ml_features['days_to_sanction'].fillna(ml_features['days_to_sanction'].median())
ml_features['days_since_sanction'] = ml_features['days_since_sanction'].fillna(ml_features['days_since_sanction'].median())
iso_forest = IsolationForest(contamination=0.05, random_state=42)
works_clean['is_ml_anomaly'] = iso_forest.fit_predict(ml_features) == -1
print(f"Isolation Forest flagged: {works_clean['is_ml_anomaly'].sum()} rows")

# --- Total Risk Score ---
works_clean['Total_Risk_Score'] = (
    works_clean['is_cost_outlier'].astype(int) +
    works_clean['is_speed_outlier'].astype(int) +
    works_clean['is_stalled'].astype(int) +
    works_clean['is_ml_anomaly'].astype(int)
)

# --- Carry through synthetic marker columns ---
for col in ['is_synthetic_test_row','expected_should_be_flagged','expected_signals']:
    if col in works.columns:
        works_clean[col] = works_clean.index.map(
            works.set_index(works.index)[col]
        ).values
    else:
        works_clean[col] = np.nan

# Reload from the raw combined (to get the extra columns properly)
raw_combined = pd.read_csv(RAW_CSV, low_memory=False)
raw_combined.index = raw_combined.index   # keep original indexing
marker_cols = ['is_synthetic_test_row','expected_should_be_flagged','expected_signals']
# Merge markers back by Sr. No.
marker_df = raw_combined[['Sr. No.'] + marker_cols].dropna(subset=['is_synthetic_test_row'])
works_clean = works_clean.merge(marker_df, on='Sr. No.', how='left', suffixes=('_drop',''))
# Drop the _drop versions
for col in marker_cols:
    if col+'_drop' in works_clean.columns:
        works_clean.drop(columns=[col+'_drop'], inplace=True)

works_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\nTest output saved to: {OUTPUT_CSV}")
print(f"Total rows in output: {len(works_clean)}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — ACCURACY MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ACCURACY MEASUREMENT")
print("="*60)

synth = works_clean[works_clean['is_synthetic_test_row'] == True].copy()
print(f"Synthetic rows found in output: {len(synth)}")

# Ground truth vs. actual (threshold: any signal fired = Total_Risk_Score >= 1)
synth['actually_flagged'] = synth['Total_Risk_Score'] >= 1
synth['gt'] = synth['expected_should_be_flagged'].astype(bool)

tp = ((synth['gt'] == True)  & (synth['actually_flagged'] == True)).sum()
tn = ((synth['gt'] == False) & (synth['actually_flagged'] == False)).sum()
fp = ((synth['gt'] == False) & (synth['actually_flagged'] == True)).sum()
fn = ((synth['gt'] == True)  & (synth['actually_flagged'] == False)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n--- Confusion Matrix (threshold: Total_Risk_Score >= 1) ---")
print(f"  True  Positives (TP): {tp}")
print(f"  True  Negatives (TN): {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Per-signal breakdown
print("\n--- Per-Signal Recall ---")
for sig, col in [("speed_outlier","is_speed_outlier"),
                 ("cost_outlier", "is_cost_outlier"),
                 ("stalled",      "is_stalled"),
                 ("ml_anomaly",   "is_ml_anomaly")]:
    subset = synth[synth['expected_signals'].str.contains(sig, na=False)]
    if len(subset) == 0:
        print(f"  {sig}: no rows designed for this signal")
        continue
    caught = subset[col].sum()
    print(f"  {sig}: designed={len(subset)}, caught={caught}, recall={caught/len(subset):.3f}")

# Misclassified rows
print("\n--- False Negatives (designed to flag, missed) ---")
fn_rows = synth[(synth['gt'] == True) & (synth['actually_flagged'] == False)]
if len(fn_rows) == 0:
    print("  None!")
else:
    for _, r in fn_rows.iterrows():
        print(f"  Sr.No={r['Sr. No.']} | expected={r['expected_signals']} | "
              f"score={r['Total_Risk_Score']} | "
              f"speed={r['is_speed_outlier']} cost={r['is_cost_outlier']} "
              f"stalled={r['is_stalled']} ml={r['is_ml_anomaly']}")

print("\n--- False Positives (designed clean, got flagged) ---")
fp_rows = synth[(synth['gt'] == False) & (synth['actually_flagged'] == True)]
if len(fp_rows) == 0:
    print("  None!")
else:
    for _, r in fp_rows.iterrows():
        print(f"  Sr.No={r['Sr. No.']} | expected=none | "
              f"score={r['Total_Risk_Score']} | "
              f"speed={r['is_speed_outlier']} cost={r['is_cost_outlier']} "
              f"stalled={r['is_stalled']} ml={r['is_ml_anomaly']}")

# Per-signal for anomalous rows at priority threshold (>= 3)
print("\n--- Distribution of risk scores among synthetic anomalous rows ---")
anom = synth[synth['gt'] == True]
print(anom['Total_Risk_Score'].value_counts().sort_index().to_string())

print("\n--- Distribution of risk scores among synthetic CLEAN rows ---")
clean = synth[synth['gt'] == False]
print(clean['Total_Risk_Score'].value_counts().sort_index().to_string())

print("\n=== DONE ===")
print(f"\nTo remove synthetic rows from Works Sanctioned.csv later:")
print("""  python3 -c "import pandas as pd; df=pd.read_csv('Works Sanctioned.csv',low_memory=False); df[df['is_synthetic_test_row']!=True].drop(columns=['is_synthetic_test_row','expected_should_be_flagged','expected_signals'],errors='ignore').to_csv('Works Sanctioned.csv',index=False); print('Done,',len(df[df['is_synthetic_test_row']!=True]),'rows remain')" """)
