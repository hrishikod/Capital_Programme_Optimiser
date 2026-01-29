import mlflow
import numpy as np


# @mlflow.trace(name="calculate_pv_coefficients", span_type="calculation")
def calculate_pv_coefficients(
    variants: dict,
    kernels_by_dim: dict,
    allowed_starts: dict,
    start_fy: int,
    years: int,
    # discount_rate argument removed/ignored in favor of MBCM standard
    dim: str = "Total",
):
    with mlflow.start_span(name="calculate_pv_coefficients", span_type="TOOL") as span:
        pv_map = {}

        # MBCM Piecewise Discounting Schedule
        # 2.0% for first 30 years, 1.5% thereafter
        r1 = 0.02
        r2 = 0.015
        switch_year = 30

        disc_vec = np.zeros(years)
        for t in range(years):
            if t <= switch_year:
                disc_vec[t] = (1.0 + r1) ** t
            else:
                disc_vec[t] = ((1.0 + r1) ** switch_year) * ((1.0 + r2) ** (t - switch_year))

        for v, starts in allowed_starts.items():
            ker = kernels_by_dim.get(dim, {}).get(v, [])
            if not ker:
                continue

            for s in starts:
                # Calculate PV if project v starts at s
                # Kernel is aligned with project duration.
                # We need to shift it by s and discount it.
                val = 0.0
                for k, f in enumerate(ker):
                    t = s + k
                    if 0 <= t < years:
                        val += float(f) / float(disc_vec[t])

                if val != 0.0:
                    pv_map[(v, s)] = val

        span.set_attribute("output_size", len(pv_map))
        span.set_attribute("mbcm_discount_r1", r1)
        span.set_attribute("mbcm_discount_r2", r2)
        span.set_attribute("mbcm_discount_switch_year", switch_year)
        span.set_attribute("start_fy", start_fy)
        span.set_attribute("years", years)
        span.set_attribute("dim", dim)

        return pv_map
