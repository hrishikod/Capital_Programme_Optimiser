"""Command line interface for the Capital Programme Optimiser."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from capital_programme_optimiser.optimisation import solver_core
from capital_programme_optimiser.dashboard import builder as dashboard_builder
from capital_programme_optimiser.tools.envelope import (
    compute_envelope,
    format_envelope_result,
    load_envelope_selection,
)
from capital_programme_optimiser.tools.renamer import prefix_cache_files


def _parse_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def cmd_run_solver(args: argparse.Namespace) -> None:
    cost_types = _parse_list(args.cost_types) or list(solver_core.COST_TYPES_RUN)
    scenario_keys = _parse_list(args.scenarios) or list(solver_core.BENEFIT_SCENARIOS.keys())
    dims = _parse_list(args.primary_dims) or list(solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN)

    previous_dims = list(solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN)
    solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN[:] = dims
    original_run_flag = solver_core.RUN_BENEFIT_PLUSMINUS
    if args.disable_buffers:
        solver_core.RUN_BENEFIT_PLUSMINUS = False

    try:
        for cost_type in cost_types:
            if cost_type not in solver_core.COST_TYPES_RUN:
                print(f"[warn] Cost type '{cost_type}' not in configuration; proceeding anyway.")
            for key in scenario_keys:
                if key not in solver_core.BENEFIT_SCENARIOS:
                    raise ValueError(f"Unknown scenario key: {key}")
                sheet = solver_core.BENEFIT_SCENARIOS[key]
                print(f"\n=== Running {cost_type} / {key} ({sheet}) ===")
                solver_core.optimise_family_for(cost_type, key, sheet)
    finally:
        solver_core.PRIMARY_OBJECTIVE_DIMS_TO_RUN[:] = previous_dims
        solver_core.RUN_BENEFIT_PLUSMINUS = original_run_flag


def cmd_build_dashboard(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir) if args.cache_dir else dashboard_builder.CACHE_DIR
    output_dir = Path(args.output_dir) if args.output_dir else dashboard_builder.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = dashboard_builder.load_results(cache_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    filename = args.filename or f"Board_Ready_Capital_Dashboard_{timestamp}.xlsx"
    output_file = output_dir / filename
    dashboard_builder.build_dashboard(results, output_file)
    print(f"Dashboard written to {output_file}")


def cmd_envelope(args: argparse.Namespace) -> None:
    if args.selection:
        selection_path = Path(args.selection)
        selection = load_envelope_selection(selection_path)
    else:
        selection = load_envelope_selection()

    result = compute_envelope(
        selection,
        baseline_total_b=args.baseline_total,
        baseline_years=args.baseline_years,
    )
    print(format_envelope_result(result))


def cmd_rename_cache(args: argparse.Namespace) -> None:
    folder = Path(args.folder) if args.folder else None
    stats = prefix_cache_files(prefix=args.prefix, folder=folder, dry_run=args.dry_run)
    print(
        "Renamed: {renamed}\nAlready prefixed: {already_prefixed}\nTarget exists: {target_exists}".format(**stats)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capital Programme Optimiser CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-solver", help="Execute optimisation scenarios")
    p_run.add_argument("--cost-types", help="Comma separated list of cost types to run")
    p_run.add_argument("--scenarios", help="Comma separated list of benefit scenario keys to run")
    p_run.add_argument("--primary-dims", help="Comma separated list of primary objective dimensions to run")
    p_run.add_argument("--disable-buffers", action="store_true", help="Disable benefit plus/minus buffer runs")
    p_run.set_defaults(func=cmd_run_solver)

    p_dash = sub.add_parser("build-dashboard", help="Generate the Excel dashboard from cache pickles")
    p_dash.add_argument("--cache-dir", help="Alternative cache directory")
    p_dash.add_argument("--output-dir", help="Destination directory for the workbook")
    p_dash.add_argument("--filename", help="Output filename (defaults to timestamped)")
    p_dash.set_defaults(func=cmd_build_dashboard)

    p_env = sub.add_parser("envelope", help="Compute buffered annual envelope")
    p_env.add_argument("--baseline-total", type=float, default=115.0, help="Baseline total budget ($B)")
    p_env.add_argument("--baseline-years", type=int, default=50, help="Number of baseline years")
    p_env.add_argument("--selection", help="Path to YAML file with project include flags")
    p_env.set_defaults(func=cmd_envelope)

    p_ren = sub.add_parser("rename-cache", help="Prefix scenario cache pickles")
    p_ren.add_argument("--prefix", required=True, help="Prefix to apply to cache files")
    p_ren.add_argument("--folder", help="Cache folder (defaults to configured cache dir)")
    p_ren.add_argument("--dry-run", action="store_true", help="Report without renaming")
    p_ren.set_defaults(func=cmd_rename_cache)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
