from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from capital_programme_optimiser.config import Settings


@dataclass
class OptimisationContext:
    settings: Settings

    @property
    def cfg(self) -> Dict[str, float]:
        opt = self.settings.optimisation
        return {
            "YEARS": opt.years,
            "START_FY": opt.start_fy,
            "MAX_STARTS": opt.max_starts,
            "CASH_PV_RATE": opt.cash_pv_rate,
        }

    @property
    def cache_dir(self) -> Path:
        return self.settings.cache_dir()

    @property
    def scoring_workbook(self) -> Path:
        return self.settings.scoring_workbook()

    @property
    def cost_types(self):
        return self.settings.optimisation.cost_types

    @property
    def benefit_scenarios(self):
        return self.settings.data.benefit_scenarios

    @property
    def forced_start(self):
        return self.settings.forced_start

