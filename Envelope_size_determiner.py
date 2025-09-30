# --- Standalone snippet: sum P95 Nominal for INCLUDED projects, then
#     compute buffered years + annual envelope -----------------------------

from pathlib import Path
import re
import numpy as np
import pandas as pd

# >>> 1) Paste/keep your FORCED_START here (INCLUDE flags drive selection)
FORCED_START: Dict[str, Dict] = {
    "Project 1":  {"start": None, "include": True},
    "Project 2":  {"start": None, "include": True},
    "Project 3":  {"start": None, "include": True},
    "Project 4":  {"start": None, "include": True},
    "Project 5":  {"start": None, "include": True},
    "Project 6":  {"start": None, "include": True},
    "Project 7":  {"start": None, "include": True},
    "Project 8":  {"start": None, "include": True},
    "Project 9":  {"start": None, "include": True},
    "Project 10": {"start": None, "include": True},
    "Project 11": {"start": None, "include": True},
    "Project 12": {"start": None, "include": True},
    "Project 13": {"start": None, "include": True},
    "Project 14": {"start": None, "include": True},
    "Project 15": {"start": None, "include": True},
    "Project 16": {"start": None, "include": True},
    "Project 17": {"start": None, "include": True},
    "Project 18": {"start": None, "include": True},
    "Project 19": {"start": None, "include": True},
    "Project 20": {"start": None, "include": True},
    "Project 21": {"start": None, "include": True},
    "Project 22": {"start": None, "include": True},
    "Project 23": {"start": None, "include": True},
    "Project 24": {"start": None, "include": True},
    "Project 25": {"start": None, "include": True},
    "Project 26": {"start": None, "include": True},
    "Project 27": {"start": None, "include": True},
    "Project 28": {"start": None, "include": True},
    "Project 29": {"start": None, "include": True},
    "Project 30": {"start": None, "include": True},
    "Project 31": {"start": None, "include": True},
    "Project 32": {"start": None, "include": True},
    "Project 33": {"start": None, "include": True},
    "Project 34": {"start": None, "include": True},
    "Project 35": {"start": None, "include": True},
    "Project 36": {"start": None, "include": True},
}

# >>> 2) Point to your workbook (Costs sheet must contain P95 Nominal rows)
DATA_FILE = Path(r"C:\Users\Adrian Desilvestro\Documents\NZTA\Project_Rons_optimisation\Scoring_latest.xlsx")
COSTS_SHEET = "Costs"

# >>> 3) Baseline capacity assumption (example from your note)
BASELINE_TOTAL_B = 115.0   # $ billions fits in...
BASELINE_YEARS   = 50      # ...this many years

# ---------- helpers ----------
def _norm_name(s) -> str:
    if s is None:
        return ""
    txt = str(s).replace("\xa0", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ---------- load costs (P95 Nominal) ----------
df = pd.read_excel(DATA_FILE, sheet_name=COSTS_SHEET, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]

# identify columns
project_col = next((c for c in df.columns if c.lower() == "project"), None)
ct_col      = next((c for c in df.columns if c.lower() == "cost type"), None)
if project_col is None or ct_col is None:
    raise RuntimeError("Expected 'Project' and 'Cost type' columns in Costs sheet.")

year_cols = sorted([c for c in df.columns if str(c).isdigit()], key=int)
if not year_cols:
    raise RuntimeError("No year columns (numeric headers) found in Costs sheet.")

# filter to P95 - Nominal; if absent, fallback to any 'P95'
mask_p95_nom = df[ct_col].astype(str).str.contains("P95", case=False, regex=False) & \
               df[ct_col].astype(str).str.contains("Nominal", case=False, regex=False)
if not mask_p95_nom.any():
    mask_p95_nom = df[ct_col].astype(str).str.contains("P95", case=False, regex=False)

cut = df.loc[mask_p95_nom].copy()

# normalise names and compute project totals ($)
cut["Project_norm"] = cut[project_col].map(_norm_name)
for c in year_cols:
    cut[c] = pd.to_numeric(cut[c], errors="coerce").fillna(0.0)

# collapse possible duplicate rows per project (sum across variants if present)
totals_by_project = cut.groupby("Project_norm")[year_cols].sum().sum(axis=1)  # $
totals_by_project = totals_by_project[totals_by_project > 0]  # drop zero rows

# ---------- choose INCLUDED set ----------
include_map = { _norm_name(k): (v.get("include") if isinstance(v, dict) else None)
                for k, v in FORCED_START.items() }

include_true = {p for p, inc in include_map.items() if inc is True}
exclude_true = {p for p, inc in include_map.items() if inc is False}

whitelist_mode = len(include_true) > 0
all_p95_projects = set(totals_by_project.index)

if whitelist_mode:
    kept = all_p95_projects & include_true
else:
    kept = all_p95_projects - exclude_true

if not kept:
    raise RuntimeError("No projects selected after INCLUDE/FALSE logic for P95 Nominal.")

# ---------- compute included totals ----------
included_series = totals_by_project.loc[sorted(kept)]
included_total_nzd = float(included_series.sum())
included_total_B   = included_total_nzd / 1e9
included_total_M   = included_total_nzd / 1e6

# ---------- 'clever' buffer for years (leniency for fewer projects & concentration) ----------
baseline_n = len(all_p95_projects)
included_n = len(kept)

# scarcity slack (fewer projects → less packing freedom), up to +25%
scarcity = max(0.0, (baseline_n - included_n) / max(1.0, baseline_n))
slack1 = 1.0 + 0.25 * scarcity

# concentration slack via normalized HHI, up to +15% (peaky = harder to pack)
shares = (included_series / included_series.sum()).to_numpy(dtype=float)
hhi = float(np.sum(shares**2))
hhi_min = 1.0 / max(1, included_n)
conc_norm = (hhi - hhi_min) / (1.0 - hhi_min) if included_n > 1 else 1.0
slack2 = 1.0 + 0.15 * conc_norm

buffer_mult = min(1.5, slack1 * slack2)  # cap total slack at +50%

# proportional years, then apply buffer & ceil to integer years
raw_years   = BASELINE_YEARS * (included_total_B / BASELINE_TOTAL_B)
years_needed = max(1, int(np.ceil(raw_years * buffer_mult)))

annual_envelope_B = included_total_B / years_needed
annual_envelope_M = included_total_M / years_needed

# ---------- report ----------
print("=== P95 Nominal (Included Projects) ===")
print(f"Projects kept: {included_n} / {baseline_n} (whitelist mode: {whitelist_mode})")
print(f"Total P95 Nominal: {included_total_B:,.3f} $B ({included_total_M:,.0f} M)")
print(f"Baseline capacity: {BASELINE_TOTAL_B:.1f} B over {BASELINE_YEARS} years")
print(f"Buffer multiplier: {buffer_mult:.3f}  (scarcity={slack1:.3f}, concentration={slack2:.3f})")
print(f"Raw proportional years: {raw_years:.2f}")
print(f"→ Years needed (buffered, ceil): {years_needed}")
print(f"→ Annual envelope: {annual_envelope_B:,.3f} $B/yr ({annual_envelope_M:,.0f} M $/yr)")

# Optional: see which projects are included and their totals
# for p, v in included_series.sort_values(ascending=False).items():
#     print(f"{p}: {v/1e9:,.3f} B")
