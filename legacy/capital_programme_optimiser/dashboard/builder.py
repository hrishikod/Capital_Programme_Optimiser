"""Dashboard builder extracted from legacy notebook."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name

from capital_programme_optimiser.config import load_settings
from capital_programme_optimiser.dashboard.data import (
    load_results,
    prepare_dashboard_data,
    dim_short,
    unique_ints,
    comp_tag,
)

SETTINGS = load_settings()
ROOT = SETTINGS.root
CACHE_DIR = SETTINGS.cache_dir()
OUTPUT_DIR = SETTINGS.dashboard_output_dir()
NZTA_BLUE = "#19456B"
DEFAULT_DIMENSION_NAMES = [
    "Total",
    "Healthy and safe people",
    "Inclusive access",
    "Environmental sustainability",
    "Economic Prosperity",
    "Urban Development",
    "Resilience and Security",
]

def A1(r, c, abs_row=True, abs_col=True):
    return xl_rowcol_to_cell(r, c, abs_row, abs_col)

def _rgb(hexstr):
    hexstr = hexstr.strip().lstrip("#")
    return tuple(int(hexstr[i:i+2], 16) for i in (0,2,4))

def _tint(hexstr, frac):
    r,g,b = _rgb(hexstr)
    f = max(0.0, min(1.0, frac))
    R = int(round(r*(1-f) + 255*f)); G = int(round(g*(1-f) + 255*f)); B = int(round(b*(1-f) + 255*f))
    return f"#{R:02X}{G:02X}{B:02X}"

def cm2px(cm: float) -> int:
    return int(round(cm * 37.7952755906))
SMALL_W = cm2px(10); SMALL_H = cm2px(10); WIDE_W = cm2px(25); WIDE_H = cm2px(10)

# --------------------- build workbook --------------------------------
def build_dashboard(res_dict, outfile: Path):
    dashboard_data = prepare_dashboard_data(res_dict)

    auto_prefixes = dashboard_data.auto_prefixes
    auto_labels = dashboard_data.auto_labels

    if auto_prefixes:
        print(f"Detected comparison prefixes: {auto_prefixes}")
    else:
        print("No comparison prefixes detected; treating all scenarios as optimised/base.")

    df_scen = dashboard_data.scenarios
    df_cf = dashboard_data.cf
    df_ben = dashboard_data.benefit
    df_bendim = dashboard_data.benefit_dim
    df_sched = dashboard_data.schedule
    df_spmat = dashboard_data.spend_matrix
    df_cmp = dashboard_data.comparison_pairs

    START_FY = dashboard_data.start_fy
    MODEL_YEARS = dashboard_data.model_years
    BENEFIT_RATE = dashboard_data.benefit_rate

    years = dashboard_data.years
    dims_list_ui = dashboard_data.dims
    proj_list = dashboard_data.projects
    years_list = years

# ====================== write workbook =============================
    with pd.ExcelWriter(outfile, engine="xlsxwriter",
                        datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:

        wb = writer.book
        try:
            wb.set_calc_mode('automatic')
        except Exception:
            pass

        # ---------- formats -------------------------------------------
        fmt_title  = wb.add_format({"bold": True, "font_size": 18, "font_color": "#0A2E5E", "align": "center"})
        bold       = wb.add_format({"bold": True})
        fmt_int    = wb.add_format({"num_format": "#,##0", "align": "center"})
        small      = wb.add_format({"font_size": 8})
        selector_fmt = wb.add_format({"bg_color": "#FDE9D9", "border": 1})
        gantt_cell_base = wb.add_format({"font_size": 8, "align": "center", "valign": "vcenter"})
        cap_over_fmt = wb.add_format({"bg_color": "#FFD7D7", "font_color": "#9C0006"})

        # ---------- helpers to write safe tables & names --------------
        def _ensure_nonempty(df: pd.DataFrame, template_row: dict) -> pd.DataFrame:
            if len(df) > 0: return df
            return pd.DataFrame([template_row])[list(template_row.keys())]

        def _add_table(ws, r0, c0, df: pd.DataFrame, name: str):
            ws.write_row(r0, c0, list(df.columns), bold)
            for i, row in enumerate(df.itertuples(index=False), start=r0+1):
                ws.write_row(i, c0, row)
            r1 = r0 + max(1, len(df))
            ws.add_table(r0, c0, r1, c0 + len(df.columns) - 1,
                         {'name': name, 'columns': [{'header': c} for c in df.columns]})
            return r1

        # ---------------- README --------------------------------------
        wsR = wb.add_worksheet("README"); writer.sheets["README"] = wsR
        readme_lines = [
            ["Purpose", "This workbook presents the capital programme with readable scenarios and traceable data."],
            ["How to Use", "Selectors on 'Dashboard' drive Optimised vs Comparison views. Charts update automatically."],
            ["Traceability", "Every scenario has a readable ScenarioCode and Title. Original cache file names are kept in DB_Index."],
            ["Database Sheets",
             "DB_Index (scenario catalog); DB_Pairs (compare mapping); DB_Facts (all), DB_OPT (Optimised only), DB_CMP (Comparison only), DB_Spend (per-project spend matrix), DB_Lists (selectors), DB_Mapping (legend & decode)."],
            ["Tables", "SCEN, CMP, CF, BEN, BENDIM, SCHED are table names used by formulas. They are stable even if sheets move."],
            ["ScenarioCode", "Example: 'P50-A45-Fix-4500-OBJRES' or 'Jimmy/P95-B45-Buf-+/-100-@4500-OBJTOT' (comparison prefixed)."],
            ["ScenarioTitle", "Human-readable string, e.g., 'P50 costs | A45 benefits | Fixed envelope $4,500m p.a. | Objective: Resilience and Security'."],
        ]
        wsR.set_column(0, 0, 20); wsR.set_column(1, 1, 120)
        wsR.write(0,0,"Board-Ready Capital Dashboard", fmt_title)
        for i,(k,v) in enumerate(readme_lines, start=2):
            wsR.write(i,0,k, bold); wsR.write(i,1,v)

        # ---------- DB_Lists (selectors) ------------------------------
        wsL = wb.add_worksheet("DB_Lists"); writer.sheets["DB_Lists"] = wsL
        wsL.write_row(0, 0, ["Year","Conf","Buffer","Envelope","OnOff","BenSteep","BenHorizon","BenRate","ObjDim"], bold)
        wsL.write_column(1, 0, [START_FY + i for i in range(MODEL_YEARS)])

        df_scen_modes = df_scen["Mode"].unique().tolist() if not df_scen.empty else []
        buffer_opts = []
        if "unconstrained" in df_scen_modes: buffer_opts.append("Unconstrained")
        if "fixed" in df_scen_modes:         buffer_opts.append("Fixed")
        buffer_opts += [f"+/-{b}" for b in unique_ints(df_scen["Buffer"])]
        if "CashPlus" in df_scen.columns: buffer_opts += [f"Cash +{c}" for c in unique_ints(df_scen["CashPlus"])]

        env_uniques = unique_ints(df_scen["Envelope"])
        conf_list   = sorted(df_scen["Conf"].dropna().astype(str).unique().tolist())
        steep_list  = sorted(df_scen["BenSteep"].dropna().astype(str).unique().tolist() or ["A","B"])
        horiz_list  = sorted(unique_ints(df_scen["BenHorizon"]) or [45,60])

        wsL.write_column(1, 1, conf_list or [""])
        wsL.write_column(1, 2, buffer_opts or [""])
        wsL.write_column(1, 3, env_uniques or [""])
        wsL.write_column(1, 4, ["Off","On"])
        wsL.write_column(1, 5, steep_list)
        wsL.write_column(1, 6, horiz_list)
        wsL.write(1, 7, BENEFIT_RATE)
        wsL.write_column(1, 8, dims_list_ui)

        def _def_list(name, col, n):
            last = 1 + max(n, 1)
            wb.define_name(name, f"=DB_Lists!${xl_col_to_name(col)}$2:${xl_col_to_name(col)}${last}")
        _def_list("Lists_Years",0,MODEL_YEARS)
        _def_list("Lists_Conf",1,len(conf_list))
        _def_list("Lists_Buffer",2,len(buffer_opts))
        _def_list("Lists_Envelope",3,len(env_uniques))
        _def_list("Lists_OnOff",4,2)
        _def_list("Lists_BenSteep",5,len(steep_list))
        _def_list("Lists_BenHorizon",6,len(horiz_list))
        _def_list("Lists_ObjDim",8,len(dims_list_ui))
        wb.define_name("BenRate","=DB_Lists!$H$2")

        # ---------- DB_Index (scenario catalog) -----------------------
        wsSI = wb.add_worksheet("DB_Index"); writer.sheets["DB_Index"] = wsSI
        scen_cols = ["Key","Conf","BenSteep","BenHorizon","Mode","EnvStr","BuffStr",
                     "Code","Envelope","Buffer","CashPlus","ObjectiveDim","ScenarioTitle",
                     "OrigStem","CacheStem","CacheFile","StartFY","HorizonYears","BenRate","IsComp"]
        _add_table(wsSI, 0, 0, _ensure_nonempty(pd.DataFrame(df_scen, columns=scen_cols),
                                                {c:"" for c in scen_cols}), "SCEN")

        # ---------- DB_Pairs (comparison mapping) ---------------------
        wsCI = wb.add_worksheet("DB_Pairs"); writer.sheets["DB_Pairs"] = wsCI
        cols_cmp = ["BaseStem","BaseCode","BaseTitle","CompStem","CompCode","CompTitle","Prefix","CompLabel","PairKey"]
        df_cmp = pd.DataFrame(comp_pairs)
        _add_table(wsCI, 0, 0, _ensure_nonempty(pd.DataFrame(df_cmp, columns=cols_cmp),
                                                {c:"" for c in cols_cmp}), "CMP")

        # ---------- DB_Facts (CF / BEN / BENDIM / SCHED) --------------
        wsD = wb.add_worksheet("DB_Facts"); writer.sheets["DB_Facts"] = wsD

        # CF
        cf_cols = ["Key","Code","Year","Spend","ClosingNet","Envelope"]
        r_cf_end = _add_table(wsD, 0, 0,
                              _ensure_nonempty(pd.DataFrame(df_cf, columns=cf_cols),
                                               {"Key":"","Code":"","Year":0,"Spend":0.0,"ClosingNet":0.0,"Envelope":0.0}),
                              "CF")

        # BEN (Total)
        ben_cols = ["Key","Code","Year","BenefitFlow"]
        r_ben0 = r_cf_end + 3
        r_ben_end = _add_table(wsD, r_ben0, 0,
                               _ensure_nonempty(pd.DataFrame(df_ben, columns=ben_cols),
                                                {"Key":"","Code":"","Year":0,"BenefitFlow":0.0}),
                               "BEN")

        # BENDIM
        bdim_cols = ["Key","Code","Dimension","Year","BenefitFlow"]
        r_bdim0 = r_ben_end + 3
        _add_table(wsD, r_bdim0, 0,
                   _ensure_nonempty(pd.DataFrame(df_bendim, columns=bdim_cols),
                                    {"Key":"","Code":"","Dimension":"", "Year":0,"BenefitFlow":0.0}),
                   "BENDIM")

        # Names bound to table columns
        wb.define_name("CF_Keys",        "=CF[Key]")
        wb.define_name("CF_Spend",       "=CF[Spend]")
        wb.define_name("CF_ClosingNet",  "=CF[ClosingNet]")
        wb.define_name("CF_Envelope",    "=CF[Envelope]")

        wb.define_name("BEN_Keys",       "=BEN[Key]")
        wb.define_name("BEN_Flow",       "=BEN[BenefitFlow]")

        wb.define_name("BENDIM_Keys",    "=BENDIM[Key]")
        wb.define_name("BENDIM_Code",    "=BENDIM[Code]")
        wb.define_name("BENDIM_Dim",     "=BENDIM[Dimension]")
        wb.define_name("BENDIM_Year",    "=BENDIM[Year]")
        wb.define_name("BENDIM_Flow",    "=BENDIM[BenefitFlow]")

        # SCHED
        sch_cols = ["Code","Project","StartFY","EndFY","Dur"]
        r_sch0 = r_bdim0 + (len(df_bendim) if not df_bendim.empty else 1) + 4
        _add_table(wsD, r_sch0, 0,
                   _ensure_nonempty(pd.DataFrame(df_sched, columns=sch_cols),
                                    {"Code":"","Project":"","StartFY":0,"EndFY":0,"Dur":0}),
                   "SCHED")

        # ---------- DB_OPT / DB_CMP (filtered views for audit) --------
        def _write_filtered(name_ws, is_comp_val):
            wsX = wb.add_worksheet(name_ws); writer.sheets[name_ws] = wsX
            sel_codes = set(df_scen[df_scen["IsComp"] == is_comp_val]["Code"].tolist())
            df_cf_x    = df_cf[df_cf["Code"].isin(sel_codes)]
            df_ben_x   = df_ben[df_ben["Code"].isin(sel_codes)]
            df_bdim_x  = df_bendim[df_bendim["Code"].isin(sel_codes)]
            df_sched_x = df_sched[df_sched["Code"].isin(sel_codes)]
            r0 = _add_table(wsX, 0, 0, _ensure_nonempty(df_cf_x,    {"Key":"","Code":"","Year":0,"Spend":0.0,"ClosingNet":0.0,"Envelope":0.0}), "CF_" + ("OPT" if is_comp_val==0 else "CMP"))
            r0 = _add_table(wsX, r0+3, 0, _ensure_nonempty(df_ben_x,   {"Key":"","Code":"","Year":0,"BenefitFlow":0.0}), "BEN_" + ("OPT" if is_comp_val==0 else "CMP"))
            r0 = _add_table(wsX, r0+3, 0, _ensure_nonempty(df_bdim_x,  {"Key":"","Code":"","Dimension":"", "Year":0,"BenefitFlow":0.0}), "BENDIM_" + ("OPT" if is_comp_val==0 else "CMP"))
            _add_table(wsX, r0+3, 0, _ensure_nonempty(df_sched_x, {"Code":"","Project":"","StartFY":0,"EndFY":0,"Dur":0}), "SCHED_" + ("OPT" if is_comp_val==0 else "CMP"))
        _write_filtered("DB_OPT", 0)
        _write_filtered("DB_CMP", 1)

        # ---------- DB_Spend (compact spend matrix) -------------------
        spmat_ws_name = "DB_Spend"
        wsM = wb.add_worksheet(spmat_ws_name); writer.sheets[spmat_ws_name] = wsM
        wsM.write_row(0, 0, ["Key","Code","Project"] + years_list, bold)
        for r_i, row in enumerate(df_spmat.itertuples(index=False), start=1):
            wsM.write_row(r_i, 0, row)
        last_row = max(1, len(df_spmat)) + 1
        last_col = 2 + len(years_list)
        year_first = xl_col_to_name(3)
        year_last  = xl_col_to_name(last_col)
        wb.define_name("SPMAT_Keys",   f"={spmat_ws_name}!$A$2:$A${last_row}")
        wb.define_name("SPMAT_Years",  f"={spmat_ws_name}!${year_first}$1:${year_last}$1")
        wb.define_name("SPMAT_Values", f"={spmat_ws_name}!${year_first}$2:${year_last}${last_row}")

        # ---------- DB_Mapping (legend + decode) ----------------------
        wsMap = wb.add_worksheet("DB_Mapping"); writer.sheets["DB_Mapping"] = wsMap
        wsMap.set_column(0, 0, 26); wsMap.set_column(1, 1, 120)

        wsMap.write(0,0,"Legend - Scenario Code Patterns", fmt_title)
        r = 2
        wsMap.write_row(r, 0, ["Item","Pattern","Notes"], bold); r += 1
        legend_rows = [
            ["Pattern (Unconstrained)", "[PrefixTag/]<Pxx>-<A|B><Horizon>-Unc-OBJ<DimCode>", "No annual envelope cap."],
            ["Pattern (Fixed)",         "[PrefixTag/]<Pxx>-<A|B><Horizon>-Fix-<ENV>-OBJ<DimCode>", "Fixed annual envelope ENV ($m p.a.)."],
            ["Pattern (Buffered)",      "[PrefixTag/]<Pxx>-<A|B><Horizon>-Buf-+/-<BUF>-@<ENV>-OBJ<DimCode>", "Buffered around ENV; zero floor, no borrowing."],
            ["Pattern (Cash+)",         "[PrefixTag/]<Pxx>-<A|B><Horizon>-Cash-+<CASH>-@<ENV>-OBJ<DimCode>", "Cash-plus on baseline ENV."],
        ]
        for row in legend_rows: wsMap.write_row(r, 0, row); r += 1
        r += 1

        wsMap.write(r,0,"Component Dictionary", fmt_title); r += 2
        wsMap.write_row(r, 0, ["Token","Meaning"], bold); r += 1
        comp_dict = [
            ["PrefixTag", "Auto-detected label family for comparisons."],
            ["P50 / P95", "Cost confidence level."],
            ["A / B",     "Benefit-flow series."],
            ["<Horizon>", "Benefit horizon in years."],
            ["Unc",       "Unconstrained envelope."],
            ["Fix-<ENV>", "Fixed envelope <ENV> $m p.a."],
            ["Buf-+/-<BUF>", "Buffered: +/-<BUF> $m above baseline (per year)."],
            ["@<ENV>",    "Baseline envelope ($m p.a.)."],
            ["Cash-+<CASH>", "Adds +<CASH> $m on top of baseline."],
            ["OBJ<DimCode>", "Primary objective dimension."],
        ]
        for row in comp_dict: wsMap.write_row(r, 0, row); r += 1
        r += 1

        wsMap.write(r,0,"Comparison Prefix Map", fmt_title); r += 2
        tag_map = {}
        for pref in auto_prefixes:
            lbl = auto_labels.get(pref, "")
            tag_map[comp_tag(pref, lbl)] = lbl or "Comparison"
        tag_rows = sorted([[k, v] for k,v in tag_map.items()], key=lambda x: x[0]) or [["-","-"]]
        wsMap.write_row(r, 0, ["PrefixTag","Comparison Label"], bold); r += 1
        for row in tag_rows: wsMap.write_row(r, 0, row); r += 1
        r += 1

        wsMap.write(r,0,"Dimension Code Map", fmt_title); r += 2
        dim_names = sorted(set(DEFAULT_DIMENSION_NAMES) | set(dims_all))
        wsMap.write_row(r, 0, ["Token","Short Code","Dimension Name"], bold); r += 1
        for nm in dim_names:
            code = dim_short(nm)
            wsMap.write_row(r, 0, [f"OBJ{code}", code, nm]); r += 1
        r += 1

        # ---------- _Calc (helper names/formulas) ---------------------
        wsC = wb.add_worksheet("_Calc"); writer.sheets["_Calc"] = wsC
        names = {}; rptr = 1
        def put(name, formula):
            nonlocal rptr
            wsC.write(rptr, 0, name); wsC.write_formula(rptr, 1, formula)
            wb.define_name(name, f"=_Calc!{A1(rptr,1)}"); names[name] = rptr; rptr += 1

        # Bind to Dashboard selectors (same cells)
        put("SelConf",        "=Dashboard!$B$1")
        put("SelBenSteep",    "=Dashboard!$B$9")
        put("SelBenHorizon",  "=Dashboard!$B$10")
        put("SelObjDim",      "=Dashboard!$B$11")

        put("OptMode",
            '=IF(Dashboard!$B$3="Unconstrained","unconstrained",'
            'IF(Dashboard!$B$3="Fixed","fixed",'
            'IF(LEFT(Dashboard!$B$3,3)="+/-","buffered",'
            'IF(LEFT(Dashboard!$B$3,4)="Cash","cash",""))))')
        put("OptBufferValue",
            '=IF(LEFT(Dashboard!$B$3,3)="+/-",VALUE(SUBSTITUTE(Dashboard!$B$3,"+/-","")),' 
            'IF(LEFT(Dashboard!$B$3,4)="Cash",VALUE(SUBSTITUTE(SUBSTITUTE(Dashboard!$B$3,"Cash +","")," ","")),""))')
        put("OptEnvStr",  '=IF(OptMode="unconstrained","",TEXT(Dashboard!$B$2,"0"))')
        put("OptBuffStr", '=IF(OptMode="buffered","+/-"&TEXT(OptBufferValue,"0"),IF(OptMode="cash","cash+"&TEXT(OptBufferValue,"0"),""))')
        put("OptKey",     '=SelConf&"|"&SelBenSteep&"|"&SelBenHorizon&"|"&OptMode&"|"&OptEnvStr&"|"&OptBuffStr)')

        # Prefer IsComp=0; fall back to any matching scenario
        put("SelCode_Opt",
            '=IFERROR(XLOOKUP(1,(SCEN[Key]=OptKey)*(SCEN[IsComp]=0),SCEN[Code],""),'
            ' IFERROR(XLOOKUP(OptKey,SCEN[Key],SCEN[Code],""),""))')

        put("CompMode",
            '=IF(Dashboard!$B$8="Unconstrained","unconstrained",'
            'IF(Dashboard!$B$8="Fixed","fixed",'
            'IF(LEFT(Dashboard!$B$8,3)="+/-","buffered",'
            'IF(LEFT(Dashboard!$B$8,4)="Cash","cash",""))))')
        put("CompBufferValue",
            '=IF(LEFT(Dashboard!$B$8,3)="+/-",VALUE(SUBSTITUTE(Dashboard!$B$8,"+/-","")),' 
            'IF(LEFT(Dashboard!$B$8,4)="Cash",VALUE(SUBSTITUTE(SUBSTITUTE(Dashboard!$B$8,"Cash +","")," ","")),""))')
        put("CompEnvStr",  '=IF(CompMode="unconstrained","",TEXT(Dashboard!$B$7,"0"))')
        put("CompBuffStr", '=IF(CompMode="buffered","+/-"&TEXT(CompBufferValue,"0"),IF(CompMode="cash","cash+"&TEXT(CompBufferValue,"0"),""))')
        put("CompKey",     '=SelConf&"|"&SelBenSteep&"|"&SelBenHorizon&"|"&CompMode&"|"&CompEnvStr&"|"&CompBuffStr)')
        put("SelCode_Comp",
            '=IFERROR(XLOOKUP(1,(SCEN[Key]=CompKey)*(SCEN[IsComp]=1),SCEN[Code],""),'
            ' IFERROR(XLOOKUP(CompKey,SCEN[Key],SCEN[Code],""),""))')

        put("SelCompLabel",
            '=IFERROR(XLOOKUP(SelCode_Opt&"|"&SelCode_Comp,CMP[PairKey],CMP[CompLabel],"Comparison"),"Comparison")')

        # Names using GanttScenario will be defined after we place the Gantt toggle cell

        wsC.hide()

        # =================================================================
        #  Dashboard (selectors; cost-only Gantt)
        # =================================================================
        ws = wb.add_worksheet("Dashboard"); writer.sheets["Dashboard"] = ws
        ws.merge_range(1, 3, 1, 42, "Capital Expenditure Programme Dashboard", fmt_title)

        # Selectors (A1:B12) - top-left panel
        ws.write(0,0,"Confidence:",bold);                ws.write(1,0,"Envelope (Optimised):",bold)
        ws.write(2,0,"Buffer (Optimised):",bold);        ws.write(3,0,"Comparison overlay:",bold)
        ws.write(4,0,"Gantt (mirror of toggle below):",bold)  # Now MIRROR only
        ws.write(6,0,"Envelope (Comparison):",bold);     ws.write(7,0,"Buffer (Comparison):",bold)
        ws.write(8,0,"Benefit steepness (A/B):",bold);   ws.write(9,0,"Benefit horizon (years):",bold)
        ws.write(10,0,"Objective dimension:",bold);      ws.write(11,0,"Dims area chart scenario:",bold)

        default_conf  = (sorted(df_scen["Conf"].unique().tolist())[0] if not df_scen.empty else "")
        env_vals      = unique_ints(df_scen["Envelope"]); default_env = env_vals[0] if env_vals else ""
        default_buffer= ("Unconstrained" if "unconstrained" in (df_scen["Mode"].unique().tolist() if not df_scen.empty else [])
                         else (["+/-0"] + buffer_opts)[0] if buffer_opts else "")
        default_steep = (df_scen["BenSteep"].dropna().astype(str).unique().tolist() or ["A"])[0]
        default_horiz = (sorted(unique_ints(df_scen["BenHorizon"]) or [45,60]))[0]
        default_dim   = "Total" if "Total" in dims_list_ui else (dims_list_ui[0] if dims_list_ui else "Total")

        ws.write(0,1, default_conf, selector_fmt); ws.data_validation(0,1,0,1, {"validate":"list","source":"=Lists_Conf"})
        ws.write(1,1, default_env,  selector_fmt); ws.data_validation(1,1,1,1, {"validate":"list","source":"=Lists_Envelope"})
        ws.write(2,1, default_buffer, selector_fmt); ws.data_validation(2,1,2,1, {"validate":"list","source":"=Lists_Buffer"})
        ws.write(3,1, "On", selector_fmt); ws.data_validation(3,1,3,1, {"validate":"list","source":"=Lists_OnOff"}); wb.define_name("CmpSwitch","=Dashboard!$B$4")
        # Cell B5 will be filled later to mirror the Gantt-area toggle (display-only)

        ws.write(6,1, default_env, selector_fmt); ws.data_validation(6,1,6,1, {"validate":"list","source":"=Lists_Envelope"})
        ws.write(7,1, default_buffer, selector_fmt); ws.data_validation(7,1,7,1, {"validate":"list","source":"=Lists_Buffer"})
        ws.write(8,1, default_steep, selector_fmt); ws.data_validation(8,1,8,1, {"validate":"list","source":"=Lists_BenSteep"})
        ws.write(9,1, default_horiz, selector_fmt); ws.data_validation(9,1,9,1, {"validate":"list","source":"=Lists_BenHorizon"})
        ws.write(10,1, default_dim, selector_fmt); ws.data_validation(10,1,10,1, {"validate":"list","source":"=Lists_ObjDim"})
        ws.write(11,1, "Optimised", selector_fmt); ws.data_validation(11,1,11,1, {"validate":"list","source":["Optimised","Comparison"]})

        ws.write(5, 5, "Overlay label:", bold); ws.write_formula(5, 6, "=SelCompLabel")

        # Helper block
        helper_c = MODEL_YEARS + 12
        ws.write_row(3, helper_c,
            ["Year","Spend","Closing Net","Envelope",
             "BenefitFlow","PV Flow","PV Benefit to date","Cum Spend","Cum Benefit",
             "Spend (Cmp)","Closing Net (Cmp)","Envelope (Cmp)",
             "BenefitFlow (Cmp)","PV Flow (Cmp)","PV Benefit to date (Cmp)",
             "Cum Spend (Cmp)","Cum Benefit (Cmp)"],
            wb.add_format({"bold": True, "bottom": 2}))
        wb.define_name("Help_Years", f"=Dashboard!{A1(4, helper_c)}:{A1(3 + MODEL_YEARS, helper_c)}")

        def _cmp_if(txt): return f'=IF(CmpSwitch="On",{txt},NA())'

        for i in range(MODEL_YEARS):
            rr = 4 + i; yr = START_FY + i; year_cell = A1(rr, helper_c)
            ws.write(rr, helper_c, yr)
            # CF
            ws.write_formula(rr, helper_c + 1, f'=XLOOKUP(SelCode_Opt&"|"&{yr},CF_Keys,CF_Spend,0)')
            ws.write_formula(rr, helper_c + 2, f'=XLOOKUP(SelCode_Opt&"|"&{yr},CF_Keys,CF_ClosingNet,0)')
            ws.write_formula(rr, helper_c + 3, f'=XLOOKUP(SelCode_Opt&"|"&{yr},CF_Keys,CF_Envelope,0)')
            # Benefits
            ws.write_formula(rr, helper_c + 4,
                f'=IFERROR(SUMIFS(BENDIM_Flow,BENDIM_Code,SelCode_Opt,BENDIM_Dim,SelObjDim,BENDIM_Year,{yr}),'
                f' XLOOKUP(SelCode_Opt&"|"&{yr},BEN_Keys,BEN_Flow,0))')
            ws.write_formula(rr, helper_c + 5, f'={A1(rr, helper_c + 4)}/POWER(1+BenRate, MATCH({year_cell},Help_Years,0)-1)')
            if i == 0:
                ws.write_formula(rr, helper_c + 6, f'={A1(rr, helper_c + 5)}')
                ws.write_formula(rr, helper_c + 7, f'={A1(rr, helper_c + 1)}')
                ws.write_formula(rr, helper_c + 8, f'={A1(rr, helper_c + 4)}')
            else:
                ws.write_formula(rr, helper_c + 6, f'={A1(rr-1, helper_c + 6)}+{A1(rr, helper_c + 5)}')
                ws.write_formula(rr, helper_c + 7, f'={A1(rr-1, helper_c + 7)}+{A1(rr, helper_c + 1)}')
                ws.write_formula(rr, helper_c + 8, f'={A1(rr-1, helper_c + 8)}+{A1(rr, helper_c + 4)}')
            # Comparison
            ws.write_formula(rr, helper_c + 9,  _cmp_if(f'XLOOKUP(SelCode_Comp&"|"&{yr},CF_Keys,CF_Spend,NA())'))
            ws.write_formula(rr, helper_c + 10, _cmp_if(f'XLOOKUP(SelCode_Comp&"|"&{yr},CF_Keys,CF_ClosingNet,NA())'))
            ws.write_formula(rr, helper_c + 11, _cmp_if(f'XLOOKUP(SelCode_Comp&"|"&{yr},CF_Keys,CF_Envelope,NA())'))
            ws.write_formula(rr, helper_c + 12,
                _cmp_if(f'IFERROR(SUMIFS(BENDIM_Flow,BENDIM_Code,SelCode_Comp,BENDIM_Dim,SelObjDim,BENDIM_Year,{yr}),'
                        f' XLOOKUP(SelCode_Comp&"|"&{yr},BEN_Keys,BEN_Flow,NA()))'))
            ws.write_formula(rr, helper_c + 13, _cmp_if(f'{A1(rr, helper_c + 12)}/POWER(1+BenRate, MATCH({year_cell},Help_Years,0)-1)'))
            if i == 0:
                ws.write_formula(rr, helper_c + 14, _cmp_if(f'{A1(rr, helper_c + 13)}'))
                ws.write_formula(rr, helper_c + 15, _cmp_if(f'{A1(rr, helper_c + 9)}'))
                ws.write_formula(rr, helper_c + 16, _cmp_if(f'{A1(rr, helper_c + 12)}'))
            else:
                ws.write_formula(rr, helper_c + 14, _cmp_if(f'{A1(rr-1, helper_c + 14)}+{A1(rr, helper_c + 13)}'))
                ws.write_formula(rr, helper_c + 15, _cmp_if(f'{A1(rr-1, helper_c + 15)}+{A1(rr, helper_c + 9)}'))
                ws.write_formula(rr, helper_c + 16, _cmp_if(f'{A1(rr-1, helper_c + 16)}+{A1(rr, helper_c + 12)}'))

        # =================================================================
        #  Charts (unchanged)
        # =================================================================
        def title14(name): return {'name': name, 'name_font': {'size': 14}}
        def legend_bottom(): return {'position': 'bottom'}
        anchor_cell = 'D6'
        def pos(x_idx, y_idx): return {'x_offset': x_idx * SMALL_W, 'y_offset': y_idx * SMALL_H}

        pv_chart = wb.add_chart({'type': 'line'}); pv_chart.set_size({'width': SMALL_W, 'height': SMALL_H})
        pv_chart.set_title(title14('Benefit PV to date ($m, real)')); pv_chart.set_legend(legend_bottom())
        pv_chart.add_series({'name':'PV Benefit to date',
                             'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                             'values':['Dashboard',4,helper_c+6,3+MODEL_YEARS,helper_c+6]})
        pv_chart.add_series({'name':['_Calc',  names["SelCompLabel"], 1],
                             'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                             'values':['Dashboard',4,helper_c+14,3+MODEL_YEARS,helper_c+14],
                             'line':{'dash_type':'dash'}})
        ws.insert_chart(anchor_cell, pv_chart, pos(0,0))

        cash_chart = wb.add_chart({'type':'column'}); cash_chart.set_size({'width':SMALL_W,'height':SMALL_H})
        cash_chart.set_title(title14('Cash-Flow Analysis - Optimised ($m)')); cash_chart.set_legend(legend_bottom())
        cash_chart.add_series({'name':'Spend',
                               'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                               'values':['Dashboard',4,helper_c+1,3+MODEL_YEARS,helper_c+1],
                               'gap':15,'fill':{'color':NZTA_BLUE},'border':{'none':True}})
        for nm, offs in (('Closing Net',2),('Envelope',3)):
            ln = wb.add_chart({'type':'line'})
            ln.add_series({'name':nm,
                           'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                           'values':['Dashboard',4,helper_c+offs,3+MODEL_YEARS,helper_c+offs]})
            cash_chart.combine(ln)
        ws.insert_chart(anchor_cell, cash_chart, pos(1,0))

        cash_chart_cmp = wb.add_chart({'type':'column'}); cash_chart_cmp.set_size({'width':SMALL_W,'height':SMALL_H})
        cash_chart_cmp.set_title(title14('Cash-Flow Analysis (Comparison) ($m)')); cash_chart_cmp.set_legend(legend_bottom())
        cash_chart_cmp.add_series({'name':['_Calc', names["SelCompLabel"], 1],
                                   'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                                   'values':['Dashboard',4,helper_c+9,3+MODEL_YEARS,helper_c+9],
                                   'gap':15,'border':{'none':True}})
        for nm, offs in (('Closing Net (Cmp)',10),('Envelope (Cmp)',11)):
            ln = wb.add_chart({'type':'line'})
            ln.add_series({'name':nm,
                           'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                           'values':['Dashboard',4,helper_c+offs,3+MODEL_YEARS,helper_c+offs],
                           'line':{'dash_type':'dash'}})
            cash_chart_cmp.combine(ln)
        ws.insert_chart(anchor_cell, cash_chart_cmp, pos(2,0))

        ben_chart = wb.add_chart({'type':'line'}); ben_chart.set_size({'width':SMALL_W,'height':SMALL_H})
        ben_chart.set_title(title14('Benefit Flow & PV Benefit-to-date ($m)')); ben_chart.set_legend(legend_bottom())
        ben_chart.add_series({'name':'BenefitFlow',
                              'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                              'values':['Dashboard',4,helper_c+4,3+MODEL_YEARS,helper_c+4]})
        ben_chart.add_series({'name':'PV Benefit-to-date',
                              'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                              'values':['Dashboard',4,helper_c+6,3+MODEL_YEARS,helper_c+6],'y2_axis':1})
        ben_chart.add_series({'name':['_Calc', names["SelCompLabel"], 1],
                              'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                              'values':['Dashboard',4,helper_c+12,3+MODEL_YEARS,helper_c+12],
                              'line':{'dash_type':'dash'}})
        ben_chart.add_series({'name':'PV Benefit-to-date (Cmp)',
                              'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                              'values':['Dashboard',4,helper_c+14,3+MODEL_YEARS,helper_c+14],
                              'line':{'dash_type':'dash'},'y2_axis':1})
        ben_chart.set_y_axis({'name':'Annual Benefit ($m)'}); ben_chart.set_y2_axis({'name':'PV Benefit-to-date ($m)'})
        ws.insert_chart(anchor_cell, ben_chart, pos(0,1))

        eff_col = wb.add_chart({'type':'column'}); eff_col.set_size({'width':SMALL_W,'height':SMALL_H})
        eff_col.set_title(title14('Efficiency Curve - Cumulative Spend vs Benefit ($m)')); eff_col.set_legend(legend_bottom())
        eff_col.add_series({'name':'Cum Spend',
                            'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
                            'values':['Dashboard',4,helper_c+7,3+MODEL_YEARS,helper_c+7],
                            'gap':10,'fill':{'color':NZTA_BLUE},'border':{'none':True}})
        eff_line = wb.add_chart({'type':'line'})
        eff_line.add_series({'name':'Cum Benefit (Optimised)',
            'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
            'values':['Dashboard',4,helper_c+8,3+MODEL_YEARS,helper_c+8]})
        eff_line_cmp = wb.add_chart({'type':'line'})
        eff_line_cmp.add_series({'name':['_Calc', names["SelCompLabel"], 1],
            'categories':['Dashboard',4,helper_c,3+MODEL_YEARS,helper_c],
            'values':['Dashboard',4,helper_c+16,3+MODEL_YEARS,helper_c+16],
            'line':{'dash_type':'dash'}})
        eff_col.combine(eff_line); eff_col.combine(eff_line_cmp)
        ws.insert_chart(anchor_cell, eff_col, pos(1,1))

        # ======= Spend-by-Project (hidden grids from DB_Spend) =========
        n_proj = len(proj_list)
        sbp_r0 = 4; sbp_c0 = helper_c + 42
        ws.write(sbp_r0 - 1, sbp_c0, "Spend by Project (hidden - base)", bold)
        for yc in range(MODEL_YEARS):
            ws.write(sbp_r0, sbp_c0 + 1 + yc, START_FY + yc, wb.add_format({"bold": True}))
        for rproj, pr in enumerate(proj_list):
            ws.write(sbp_r0 + 1 + rproj, sbp_c0, pr, small)
            for yc in range(MODEL_YEARS):
                yr = START_FY + yc
                ws.write_formula(sbp_r0 + 1 + rproj, sbp_c0 + 1 + yc,
                    f'=INDEX(SPMAT_Values, MATCH(SelCode_Opt&"|"&{A1(sbp_r0 + 1 + rproj, sbp_c0, False, False)}, SPMAT_Keys, 0), '
                    f'MATCH({yr}, SPMAT_Years, 0))', fmt_int)

        sbp2_r0 = sbp_r0 + n_proj + 8; sbp2_c0 = sbp_c0
        ws.write(sbp2_r0 - 1, sbp2_c0, "Spend by Project (hidden - comparison)", bold)
        for yc in range(MODEL_YEARS):
            ws.write(sbp2_r0, sbp2_c0 + 1 + yc, START_FY + yc, wb.add_format({"bold": True}))
        for rproj, pr in enumerate(proj_list):
            ws.write(sbp2_r0 + 1 + rproj, sbp2_c0, pr, small)
            for yc in range(MODEL_YEARS):
                yr = START_FY + yc
                ws.write_formula(sbp2_r0 + 1 + rproj, sbp2_c0 + 1 + yc,
                    f'=INDEX(SPMAT_Values, MATCH(SelCode_Comp&"|"&{A1(sbp2_r0 + 1 + rproj, sbp2_c0, False, False)}, SPMAT_Keys, 0), '
                    f'MATCH({yr}, SPMAT_Years, 0))', fmt_int)

        # =================================================================
        #  Gantt grid - COST ONLY, IFERROR->0; capacity CF on bottom row
        # =================================================================
        g0 = max(sbp2_r0 + n_proj + 8, 43) + 8
        ws.write(g0, 1, "Project Delivery Schedule (Cost only, $m)", wb.add_format({"bold": True, "font_size": 14, "font_color": "#0A2E5E"}))

        # ** Define the Gantt toggle cell as the single interactive control **
        ws.write(g0, 20, "Scenario:", bold)
        gantt_toggle_row, gantt_toggle_col = g0, 21
        # Put an initial value and validation (no formula in this cell)
        ws.write(gantt_toggle_row, gantt_toggle_col, "Comparison", selector_fmt)
        ws.data_validation(gantt_toggle_row, gantt_toggle_col, gantt_toggle_row, gantt_toggle_col,
                           {"validate": "list", "source": ["Optimised", "Comparison"]})
        # Bind name AFTER the cell exists
        wb.define_name("GanttScenario", f"=Dashboard!{A1(gantt_toggle_row, gantt_toggle_col)}")
        # Mirror into top-left selector cell B5 (display-only)
        ws.write_formula(4, 1, f"=GanttScenario", selector_fmt)  # B5 mirrors the active toggle

        # Now we can safely define helpers that depend on GanttScenario
        wsC = writer.sheets["_Calc"]
        def redefine_put(name, formula):
            r = names.get(name)
            if r is not None:
                wsC.write_formula(r, 1, formula)
            else:
                # add if missing
                last = max(names.values(), default=1)
                rr = last + 1
                wsC.write(rr, 0, name); wsC.write_formula(rr, 1, formula)
                wb.define_name(name, f"=_Calc!{A1(rr,1)}"); names[name] = rr

        redefine_put("GanttCode", '=IF(GanttScenario="Comparison",SelCode_Comp,SelCode_Opt)')
        redefine_put("OtherCode", '=IF(GanttScenario="Comparison",SelCode_Opt,SelCode_Comp)')
        wb.define_name("DimChartScenario", "=Dashboard!$B$12")
        redefine_put("DimChartCode", '=IF(DimChartScenario="Comparison",SelCode_Comp,SelCode_Opt)')

        base_px   = int(round(7 * 3.2 + 5))
        square_px = max(16, int(round(base_px * 0.8)))
        row_points = square_px * 0.75
        col_width  = (square_px - 5) / 7.0

        ws.set_column(1, 1, 34); ws.set_column(2, 1 + MODEL_YEARS, col_width)
        for yc in range(MODEL_YEARS):
            ws.write(g0 + 1, 2 + yc, START_FY + yc, wb.add_format({"rotation": 90, "bold": True, "font_size": 9, "align":"center"}))

        first_proj_row = g0 + 2
        for rproj, pr in enumerate(proj_list):
            row = first_proj_row + rproj
            ws.write(row, 1, pr, small); ws.set_row(row, row_points)
            ws.write_formula(row, MODEL_YEARS + 8,
                f'=SUM({xl_rowcol_to_cell(row,2,False,False)}:{xl_rowcol_to_cell(row,1+MODEL_YEARS,False,False)})', fmt_int)
            for yc in range(MODEL_YEARS):
                col = 2 + yc; yr = START_FY + yc
                ws.write_formula(
                    row, col,
                    f'=IFERROR(INDEX(SPMAT_Values, '
                    f' MATCH(GanttCode&"|"&{A1(row,1,False,False)}, SPMAT_Keys, 0), '
                    f' MATCH({yr}, SPMAT_Years, 0) '
                    f'), 0)',
                    gantt_cell_base)

        # Bottom lines: total spend + scenario envelope
        totals_row = first_proj_row + n_proj + 1
        cap_row    = totals_row + 1
        ws.write(totals_row, 1, "All Gantt (Spend)", bold)
        ws.write(cap_row,    1, "Market Capacity ($m p.a.)", bold)

        for yc in range(MODEL_YEARS):
            col = 2 + yc
            ws.write_formula(
                totals_row, col,
                f'=SUM({xl_rowcol_to_cell(first_proj_row, col, False, False)}:{xl_rowcol_to_cell(first_proj_row + n_proj - 1, col, False, False)})',
                fmt_int)
            yr = START_FY + yc
            ws.write_formula(
                cap_row, col,
                f'=IF(GanttScenario="Comparison",'
                f' XLOOKUP(SelCode_Comp&"|"&{yr},CF_Keys,CF_Envelope,0),'
                f' XLOOKUP(SelCode_Opt&"|"&{yr},CF_Keys,CF_Envelope,0))',
                fmt_int)

        # Conditional format: flag if spend exceeds capacity
        ws.conditional_format(cap_row, 2, cap_row, 1 + MODEL_YEARS, {
            'type': 'formula',
            'criteria': f'={xl_rowcol_to_cell(totals_row,2,False,False)}>{xl_rowcol_to_cell(cap_row,2,False,False)}',
            'format': cap_over_fmt
        })

        # Freeze panes
        ws.freeze_panes(12, 2)

        print("  - workbook built: Gantt toggle bound to one interactive cell; top selector mirrors it; no selector overwrite.")

# --------------------- run -------------------------------------------
if __name__ == "__main__":
    results = load_results(CACHE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"Board_Ready_Capital_Dashboard_{datetime.now():%Y-%m-%d_%H%M%S}.xlsx"
    build_dashboard(results, fname)
    print(f"\n[OK] Dashboard written -> {fname}")




















