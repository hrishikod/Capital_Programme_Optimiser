import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class ProjectData:
    projects: Dict[str, Dict[str, Any]]
    variants: Dict[str, Dict[str, Any]]
    costs_input_df: pd.DataFrame
    benef_df: pd.DataFrame
    ben_kernel_df: pd.DataFrame
    kernels_by_dim: Dict[str, Dict[str, List[float]]]
    dims_order: List[str]

class DataLoader:
    def __init__(self, data_file: str, start_fy: int, years: int):
        self.data_file = data_file
        self.start_fy = start_fy
        self.years = years
        self.horizon_all = [start_fy + i for i in range(years)]

    def clean(self, s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "").replace("\xa0", " ")).strip()

    def norm(self, s: str) -> str:
        return self.clean(s).lower()

    def load_costs(self, cost_type: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], pd.DataFrame]:
        df = pd.read_excel(self.data_file, sheet_name="Costs", engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        proj_col = [c for c in df.columns if c.lower() == "project"]
        if not proj_col:
            raise RuntimeError("Costs sheet needs 'Project'.")
        proj_col = proj_col[-1]

        year_cols = {int(c): c for c in df.columns if str(c).isdigit()}
        use_cols = [year_cols.get(y, None) for y in self.horizon_all]

        cut = df[df["Cost type"].astype(str).str.strip() == str(cost_type).strip()].copy()
        costs_input: Dict[str, List[float]] = {}
        for _, r in cut.iterrows():
            p = self.clean(r[proj_col])
            vals = [
                (pd.to_numeric(r[c], errors="coerce") if c is not None else 0.0)
                for c in use_cols
            ]
            costs_input[p] = (pd.Series(vals).fillna(0.0) / 1_000_000.0).tolist()  # M

        projects: Dict[str, Dict[str, Any]] = {}
        variants: Dict[str, Dict[str, Any]] = {}
        for p, seriesM in costs_input.items():
            s = pd.Series(seriesM)
            nz = s.to_numpy().nonzero()[0]
            if nz.size == 0:
                continue
            seg = s.iloc[nz.min() : nz.max() + 1].tolist()
            projects[p] = {"cost": float(sum(seg)), "dur": len(seg), "spend": seg}
            variants[p] = {
                "base": p,
                "dur": len(seg),
                "spend": seg,
                "first_year_idx": int(nz.min()),
            }

        costs_input_df = pd.DataFrame(costs_input, index=self.horizon_all).T
        costs_input_df.index.name = "Project"
        return projects, variants, costs_input_df

    def load_benefits(self, sheet: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_excel(self.data_file, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        if "Project" not in df.columns:
            raise RuntimeError("Benefits sheet needs 'Project'.")
        dim_col = None
        for c in df.columns:
            if c.lower().startswith("dimension"):
                dim_col = c
                break
        if dim_col is None:
            raise RuntimeError("Benefits sheet needs 'Dimension'.")
        if dim_col != "Dimension":
            df.rename(columns={dim_col: "Dimension"}, inplace=True)

        tcols: List[Tuple[int, str]] = []
        for c in df.columns:
            m = re.fullmatch(r"[tT]\s*\+\s*(\d+)", str(c))
            if m:
                tcols.append((int(m.group(1)), c))
        tcols.sort(key=lambda x: x[0])

        for _, c in tcols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        ben_kernel_df = df.copy()
        ben_kernel_df["Project"] = ben_kernel_df["Project"].map(self.clean)
        ben_kernel_df["Dimension"] = ben_kernel_df["Dimension"].map(self.clean)
        ben_kernel_df.set_index(["Project", "Dimension"], inplace=True)
        ben_kernel_df = ben_kernel_df[[c for _, c in tcols]]
        return df, ben_kernel_df

    def map_benefit_kernels(self, benef_df: pd.DataFrame, variants: Dict[str, dict]) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
        tcols = [
            c
            for _, c in sorted(
                [
                    (int(re.fullmatch(r"[tT]\s*\+\s*(\d+)", c).group(1)), c)
                    for c in benef_df.columns
                    if re.fullmatch(r"[tT]\s*\+\s*(\d+)", c)
                ]
            )
        ]
        df = benef_df.copy()
        df["Project_clean"] = df["Project"].map(self.clean)
        df["Dimension_clean"] = df["Dimension"].map(self.clean)

        flows_by_dim: Dict[str, Dict[str, List[float]]] = {}
        order: List[str] = []
        for _, r in df.iterrows():
            p = r["Project_clean"]
            d = r["Dimension_clean"]
            seq = r[tcols].to_numpy(dtype=float).tolist()
            flows_by_dim.setdefault(d, {})[p] = seq
            if d not in order:
                order.append(d)

        if "Total" not in flows_by_dim:
            flows_by_dim["Total"] = {}
        all_dims = [d for d in order if d.lower() != "total"]
        projs = set()
        for d in all_dims:
            projs |= set(flows_by_dim[d].keys())

        for p in projs:
            acc = None
            for d in all_dims:
                v = flows_by_dim[d].get(p)
                if v is None:
                    continue
                acc = v if acc is None else [a + b for a, b in zip(acc, v)]
            flows_by_dim["Total"][p] = acc or [0.0] * len(tcols)

        if "Total" not in order:
            order.append("Total")

        keeps = set(variants.keys())
        for d in list(flows_by_dim.keys()):
            flows_by_dim[d] = {p: seq for p, seq in flows_by_dim[d].items() if p in keeps}

        kernels_by_dim: Dict[str, Dict[str, List[float]]] = {}
        for d, mp_ in flows_by_dim.items():
            kernels_by_dim[d] = {}
            for v, meta in variants.items():
                dur = meta["dur"]
                ker = mp_.get(v, [])
                kernels_by_dim[d][v] = [0.0] * dur + [float(x) for x in ker]
        return order, kernels_by_dim

    def apply_forced_rules(
        self,
        variants: Dict[str, dict],
        rules: Dict[str, Dict],
        project_selection_mode: str = "auto",
        whitelist_fallback: bool = True
    ) -> Tuple[Dict[str, dict], Dict[str, int], bool]:
        
        v_norm2canon = {self.norm(v): v for v in variants.keys()}
        v_norm_set = set(v_norm2canon.keys())
        include_true_norm, exclude_true_norm = set(), set()
        start_map_all_norm: Dict[str, int] = {}

        for raw_name, spec in (rules or {}).items():
            pname_norm = self.norm(raw_name)
            inc = spec.get("include", None)
            st = spec.get("start", None)
            if inc is True:
                include_true_norm.add(pname_norm)
            if inc is False:
                exclude_true_norm.add(pname_norm)
            if st is not None:
                start_map_all_norm[pname_norm] = int(st)

        matched_includes_norm = include_true_norm & v_norm_set
        mode_req = (project_selection_mode or "auto").strip().lower()

        has_include_true_rules = len(include_true_norm) > 0
        use_whitelist = (mode_req == "whitelist") or (
            mode_req == "auto" and has_include_true_rules
        )

        if use_whitelist:
            keep_norm = matched_includes_norm
        else:
            if len(matched_includes_norm) == 0 and whitelist_fallback:
                keep_norm = v_norm_set - (exclude_true_norm & v_norm_set)
            else:
                keep_norm = v_norm_set - (exclude_true_norm & v_norm_set)

        keep_canon = {v_norm2canon[n] for n in keep_norm}
        kept_variants = {v: variants[v] for v in variants if v in keep_canon}
        forced_exact: Dict[str, int] = {}
        for n, yr in start_map_all_norm.items():
            if n in keep_norm:
                forced_exact[v_norm2canon[n]] = int(yr)

        return kept_variants, forced_exact, use_whitelist

    def load_all(self, cost_type: str, benefit_sheet: str, rules: Dict[str, Dict]) -> ProjectData:
        projects, variants_all, costs_input_df = self.load_costs(cost_type)
        benef_df, ben_kernel_df = self.load_benefits(benefit_sheet)
        
        # Apply rules to filter variants
        variants, forced_exact, is_whitelist = self.apply_forced_rules(variants_all, rules)
        
        # Map benefits for the kept variants
        dims_order, kernels_by_dim = self.map_benefit_kernels(benef_df, variants)
        
        return ProjectData(
            projects=projects,
            variants=variants,
            costs_input_df=costs_input_df,
            benef_df=benef_df,
            ben_kernel_df=ben_kernel_df,
            kernels_by_dim=kernels_by_dim,
            dims_order=dims_order
        )
