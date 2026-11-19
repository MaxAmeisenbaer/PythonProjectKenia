#!/usr/bin/env python3
"""Compare the Air-* and Mar-* measurement series before merging them.

Background
----------
The Air-* and Mar-* files describe the same measurement setup. Mar-* covers
the period before the equipment was moved, Air-* the period afterwards. The
two series do not overlap in time, so a direct value-by-value comparison is
impossible. This script therefore compares:

1. Full-period summary statistics and distributions.
2. A "same-season" subset: only calendar months that are present in both
   series with sufficient data coverage. This reduces (but cannot remove)
   bias caused by the different measurement periods.
3. Monthly climatologies and mean diurnal cycles.
4. A short window immediately before and after the move, to check whether
   the values jump at the transition.

Important caveat
----------------
A difference between the two series is not automatically a sensor problem:
the periods cover different years and seasons. Use the effect sizes and the
transition comparison together, not a single p-value. With tens of thousands
of samples, p-values become tiny even for irrelevant differences.

Usage
-----
    python "Additional Code/checking.py"
    python "Additional Code/checking.py" --variables temp rh ec15 wind
    python "Additional Code/checking.py" --no-plots --output-dir /tmp/air_mar
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path.home() / "Documents" / "Data kopier-Ordner"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "air_mar_comparison"

VALUE_COLUMN = "value"
EXPECTED_SAMPLES_PER_DAY = 144  # 10-minute data
CIRCULAR_VARIABLES = {"dir"}  # degrees, 0-360; linear statistics are invalid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Mar-* and Air-* time series before merging them."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the Air-*.csv / Mar-*.csv files "
             f"(default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for report, CSV tables and plots "
             f"(default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--transition-days",
        type=float,
        default=7.0,
        help="Window size in days before/after the move (default: 7)",
    )
    parser.add_argument(
        "--min-month-coverage",
        type=float,
        default=0.40,
        help="Minimum 10-minute coverage for a calendar month to enter the "
             "same-season comparison (default: 0.40)",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional subset of variables, e.g. --variables temp rh ec15",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not create the per-variable comparison figures.",
    )
    return parser.parse_args()


def _read_series(path: Path) -> pd.Series:
    """Read one CSV and return a clean, datetime-indexed value series."""
    df = pd.read_csv(path, parse_dates=["date"], na_values=["NA", "NaN", ""])
    if "date" not in df.columns or VALUE_COLUMN not in df.columns:
        raise ValueError(f"{path} must contain the columns 'date' and 'value'.")

    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce")
    df = (
        df.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
        .sort_index()
    )
    return df[VALUE_COLUMN].astype(float)


def _discover_pairs(data_dir: Path, variables: list[str] | None) -> dict[str, tuple[Path, Path]]:
    """Find all Air/Mar pairs. Returns {variable: (mar_path, air_path)}."""
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    pairs: dict[str, tuple[Path, Path]] = {}
    for air_path in sorted(data_dir.glob("Air-*.csv")):
        variable = air_path.stem[len("Air-"):]
        mar_path = data_dir / f"Mar-{variable}.csv"
        if mar_path.is_file():
            pairs[variable] = (mar_path, air_path)

    if variables:
        missing = [v for v in variables if v not in pairs]
        if missing:
            raise SystemExit(
                f"No complete Air/Mar pair found for: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(pairs))}"
            )
        pairs = {v: pairs[v] for v in variables}

    if not pairs:
        raise SystemExit(f"No Air-*.csv / Mar-*.csv pairs found in {data_dir}.")
    return dict(sorted(pairs.items()))


def _safe_relative_diff(value: float, reference: float) -> float:
    """Relative difference in percent, or NaN when the reference is ~zero."""
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) <= 1e-12:
        return float("nan")
    return 100.0 * (value - reference) / abs(reference)


def _circular_mean_deg(values: np.ndarray | pd.Series) -> float:
    """Circular mean of angles in degrees."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    radians = np.deg2rad(values)
    return float(
        np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0
    )


def _circular_resultant_length(values: np.ndarray | pd.Series) -> float:
    """Concentration of angles (0 = uniform, 1 = all identical)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    radians = np.deg2rad(values)
    return float(np.abs(np.mean(np.exp(1j * radians))))


def _angular_difference_deg(value: float, reference: float) -> float:
    """Signed shortest angular difference value - reference, in [-180, 180]."""
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return float((value - reference + 180.0) % 360.0 - 180.0)


def _summarize(series: pd.Series) -> dict:
    """Basic descriptive statistics for one subset."""
    valid = series.dropna()
    missing = int(series.isna().sum())
    if valid.empty:
        return {
            "n": 0,
            "missing": missing,
            "missing_pct": 100.0 * series.isna().mean(),
            "start": pd.NaT,
            "end": pd.NaT,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
            "zero_pct": np.nan,
            "positive_pct": np.nan,
            "positive_mean": np.nan,
        }

    positive = valid[valid > 0]
    return {
        "n": int(valid.size),
        "missing": missing,
        "missing_pct": 100.0 * series.isna().mean(),
        "start": series.index.min(),
        "end": series.index.max(),
        "mean": valid.mean(),
        "median": valid.median(),
        "std": valid.std(),
        "min": valid.min(),
        "q05": valid.quantile(0.05),
        "q25": valid.quantile(0.25),
        "q75": valid.quantile(0.75),
        "q95": valid.quantile(0.95),
        "q99": valid.quantile(0.99),
        "max": valid.max(),
        "zero_pct": 100.0 * (valid == 0).mean(),
        "positive_pct": 100.0 * (valid > 0).mean(),
        "positive_mean": positive.mean() if not positive.empty else np.nan,
    }


def _month_coverage(series: pd.Series) -> pd.Series:
    """Fraction of expected 10-minute samples per calendar month.

    The denominator is based on all year/month combinations present in the
    series, so outages in a single year do not disqualify a whole month.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(dtype=float)

    coverage = {}
    for month in range(1, 13):
        month_values = valid[valid.index.month == month]
        if month_values.empty:
            coverage[month] = 0.0
            continue
        years = sorted(set(month_values.index.year))
        expected = sum(
            pd.Period(f"{year}-{month:02d}").days_in_month * EXPECTED_SAMPLES_PER_DAY
            for year in years
        )
        coverage[month] = month_values.size / expected if expected else 0.0
    return pd.Series(coverage, dtype=float)


def _common_months(
    mar: pd.Series,
    air: pd.Series,
    min_coverage: float,
) -> list[int]:
    """Calendar months present in both series with enough data."""
    mar_cov = _month_coverage(mar)
    air_cov = _month_coverage(air)
    months = set(mar_cov.index) & set(air_cov.index)
    return sorted(
        month for month in months
        if mar_cov.get(month, 0.0) >= min_coverage
        and air_cov.get(month, 0.0) >= min_coverage
    )


def _compare_subsets(
    reference: pd.Series,
    comparison: pd.Series,
    label: str,
    circular: bool = False,
) -> dict:
    """Compare two series (reference = Mar, comparison = Air)."""
    ref = reference.dropna().to_numpy(dtype=float)
    comp = comparison.dropna().to_numpy(dtype=float)

    result = {
        f"mar_n_{label}": int(ref.size),
        f"air_n_{label}": int(comp.size),
        f"mar_mean_{label}": np.mean(ref) if ref.size else np.nan,
        f"air_mean_{label}": np.mean(comp) if comp.size else np.nan,
        f"mar_median_{label}": np.median(ref) if ref.size else np.nan,
        f"air_median_{label}": np.median(comp) if comp.size else np.nan,
        f"mean_diff_{label}": np.nan,
        f"mean_rel_diff_pct_{label}": np.nan,
        f"median_diff_{label}": np.nan,
        f"median_rel_diff_pct_{label}": np.nan,
        f"cohens_d_{label}": np.nan,
        f"cliffs_delta_{label}": np.nan,
        f"ks_stat_{label}": np.nan,
        f"ks_p_{label}": np.nan,
        f"wasserstein_{label}": np.nan,
        f"wasserstein_norm_{label}": np.nan,
        f"mar_circular_mean_{label}": np.nan,
        f"air_circular_mean_{label}": np.nan,
        f"circular_mean_diff_deg_{label}": np.nan,
        f"mar_resultant_length_{label}": np.nan,
        f"air_resultant_length_{label}": np.nan,
    }

    if ref.size < 3 or comp.size < 3:
        return result

    if circular:
        ref_circular = _circular_mean_deg(ref)
        comp_circular = _circular_mean_deg(comp)
        result[f"mar_circular_mean_{label}"] = ref_circular
        result[f"air_circular_mean_{label}"] = comp_circular
        result[f"circular_mean_diff_deg_{label}"] = _angular_difference_deg(
            comp_circular, ref_circular
        )
        result[f"mar_resultant_length_{label}"] = _circular_resultant_length(ref)
        result[f"air_resultant_length_{label}"] = _circular_resultant_length(comp)

        # Linear means/effect sizes are not meaningful for angles.
        result[f"mar_mean_{label}"] = ref_circular
        result[f"air_mean_{label}"] = comp_circular
        result[f"mean_diff_{label}"] = _angular_difference_deg(comp_circular, ref_circular)
        return result

    ref_mean = np.mean(ref)
    comp_mean = np.mean(comp)
    ref_median = np.median(ref)
    comp_median = np.median(comp)

    result[f"mean_diff_{label}"] = comp_mean - ref_mean
    result[f"mean_rel_diff_pct_{label}"] = _safe_relative_diff(comp_mean, ref_mean)
    result[f"median_diff_{label}"] = comp_median - ref_median
    result[f"median_rel_diff_pct_{label}"] = _safe_relative_diff(comp_median, ref_median)

    pooled_std = np.sqrt((np.var(ref, ddof=1) + np.var(comp, ddof=1)) / 2.0)
    if pooled_std > 1e-12:
        result[f"cohens_d_{label}"] = (comp_mean - ref_mean) / pooled_std

    # U for the comparison sample -> positive Cliff's delta means Air > Mar.
    mw = stats.mannwhitneyu(comp, ref, alternative="two-sided")
    result[f"cliffs_delta_{label}"] = 2.0 * mw.statistic / (comp.size * ref.size) - 1.0

    ks = stats.ks_2samp(ref, comp)
    result[f"ks_stat_{label}"] = float(ks.statistic)
    result[f"ks_p_{label}"] = float(ks.pvalue)

    distance = stats.wasserstein_distance(ref, comp)
    result[f"wasserstein_{label}"] = float(distance)
    if pooled_std > 1e-12:
        result[f"wasserstein_norm_{label}"] = float(distance / pooled_std)

    return result


def _profile(series: pd.Series, attribute: str, circular: bool = False) -> pd.Series:
    """Mean value by calendar month or hour of day."""
    valid = series.dropna()
    keys = getattr(valid.index, attribute)
    if circular:
        return valid.groupby(keys).apply(_circular_mean_deg)
    return valid.groupby(keys).mean()


def _profile_correlation(
    reference: pd.Series,
    comparison: pd.Series,
    attribute: str,
    min_points: int = 3,
    circular: bool = False,
) -> tuple[float, float, list[int], pd.Series, pd.Series]:
    """Correlation between two mean profiles (month or hour)."""
    ref_profile = _profile(reference, attribute, circular=circular)
    comp_profile = _profile(comparison, attribute, circular=circular)
    common = sorted(set(ref_profile.index) & set(comp_profile.index))

    if circular or len(common) < min_points:
        return np.nan, np.nan, common, ref_profile, comp_profile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, p_value = stats.pearsonr(
            ref_profile.loc[common].to_numpy(),
            comp_profile.loc[common].to_numpy(),
        )
    return float(corr), float(p_value), common, ref_profile, comp_profile


def _assess(row: dict) -> str:
    """Heuristic summary of the same-season comparison.

    The thresholds are deliberately simple and are meant as a screening aid,
    not as a formal decision rule.
    """
    if row.get("is_circular", False):
        angular = row.get("circular_mean_diff_deg_common", np.nan)
        mar_length = row.get("mar_resultant_length_common", np.nan)
        air_length = row.get("air_resultant_length_common", np.nan)
        if not np.isfinite(angular):
            return "insufficient data"
        if (
            np.isfinite(mar_length) and mar_length < 0.10
        ) or (
            np.isfinite(air_length) and air_length < 0.10
        ):
            return "angular mean unreliable (diffuse directions)"
        angular_abs = abs(angular)
        if angular_abs >= 45.0:
            return "strong angular shift - review merge"
        if angular_abs >= 20.0:
            return "moderate angular shift - review"
        return "no strong angular shift"

    rel = row.get("mean_rel_diff_pct_common", np.nan)
    cohen = row.get("cohens_d_common", np.nan)
    ks = row.get("ks_stat_common", np.nan)

    rel_abs = abs(rel) if np.isfinite(rel) else np.nan
    cohen_abs = abs(cohen) if np.isfinite(cohen) else np.nan

    strong = (
        (np.isfinite(rel_abs) and rel_abs >= 50.0)
        or (np.isfinite(cohen_abs) and cohen_abs >= 0.8)
        or (np.isfinite(ks) and ks >= 0.5)
    )
    moderate = (
        (np.isfinite(rel_abs) and rel_abs >= 20.0)
        or (np.isfinite(cohen_abs) and cohen_abs >= 0.5)
        or (np.isfinite(ks) and ks >= 0.3)
    )

    if strong:
        return "strong shift - review merge"
    if moderate:
        return "moderate shift - review"
    return "no strong shift"


def _transition_windows(
    mar: pd.Series,
    air: pd.Series,
    days: float,
) -> tuple[pd.Series, pd.Series, float]:
    """Last `days` of Mar and first `days` of Air, plus the gap in hours."""
    mar_end = mar.dropna().index.max()
    air_start = air.dropna().index.min()
    if pd.isna(mar_end) or pd.isna(air_start):
        return mar.iloc[0:0], air.iloc[0:0], float("nan")

    mar_start = mar_end - pd.Timedelta(days=days)
    air_end = air_start + pd.Timedelta(days=days)

    mar_window = mar[(mar.index >= mar_start) & (mar.index <= mar_end)]
    air_window = air[(air.index >= air_start) & (air.index <= air_end)]
    gap_hours = (air_start - mar_end).total_seconds() / 3600.0
    return mar_window, air_window, gap_hours


def _compare_variable(variable: str, mar: pd.Series, air: pd.Series, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Build one comparison row and the matching summary-statistics rows."""
    months = _common_months(mar, air, args.min_month_coverage)
    mar_common = mar[mar.index.month.isin(months)]
    air_common = air[air.index.month.isin(months)]

    mar_transition, air_transition, gap_hours = _transition_windows(
        mar, air, args.transition_days
    )

    row: dict = {
        "variable": variable,
        "is_circular": variable in CIRCULAR_VARIABLES,
        "common_months": ",".join(str(month) for month in months),
        "common_month_count": len(months),
        "transition_gap_hours": gap_hours,
        "transition_days": args.transition_days,
    }

    circular = variable in CIRCULAR_VARIABLES
    row.update(_compare_subsets(mar, air, "full", circular=circular))
    row.update(_compare_subsets(mar_common, air_common, "common", circular=circular))
    row.update(
        _compare_subsets(
            mar_transition, air_transition, "transition", circular=circular
        )
    )

    monthly_corr, monthly_p, monthly_points, _, _ = _profile_correlation(
        mar_common, air_common, "month", circular=circular
    )
    diurnal_corr, diurnal_p, diurnal_points, _, _ = _profile_correlation(
        mar_common, air_common, "hour", circular=circular
    )
    row.update({
        "monthly_corr": monthly_corr,
        "monthly_corr_p": monthly_p,
        "monthly_points": len(monthly_points),
        "diurnal_corr": diurnal_corr,
        "diurnal_corr_p": diurnal_p,
        "diurnal_points": len(diurnal_points),
    })
    row["assessment"] = _assess(row)

    summaries = []
    subset_series = {
        "mar_full": mar,
        "air_full": air,
        "mar_common": mar_common,
        "air_common": air_common,
        "mar_transition": mar_transition,
        "air_transition": air_transition,
    }
    for subset, series in subset_series.items():
        stats_row = {"variable": variable, "subset": subset}
        stats_row.update(_summarize(series))
        summaries.append(stats_row)

    return row, summaries


def _downsample(series: pd.Series, max_points: int = 5000) -> pd.Series:
    """Reduce a long series for plotting without changing the overall shape."""
    valid = series.dropna()
    if valid.size <= max_points:
        return valid
    step = int(np.ceil(valid.size / max_points))
    return valid.iloc[::step]


def _histogram_limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([v[np.isfinite(v)] for v in values if v.size])
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high <= low:
        padding = max(abs(low) * 0.05, 1e-9)
        return low - padding, high + padding
    return float(low), float(high)


def _plot_variable(
    variable: str,
    mar: pd.Series,
    air: pd.Series,
    months: list[int],
    mar_transition: pd.Series,
    air_transition: pd.Series,
    transition_days: float,
    output_dir: Path,
) -> None:
    """Create a compact 2x3 overview figure for one variable."""
    circular = variable in CIRCULAR_VARIABLES
    mar_common = mar[mar.index.month.isin(months)]
    air_common = air[air.index.month.isin(months)]

    monthly_ref = _profile(mar_common, "month", circular=circular)
    monthly_comp = _profile(air_common, "month", circular=circular)
    diurnal_ref = _profile(mar_common, "hour", circular=circular)
    diurnal_comp = _profile(air_common, "hour", circular=circular)

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    ax_full, ax_transition, ax_hist, ax_monthly, ax_diurnal, ax_box = axes.flat

    # 1. Full time series
    mar_plot = _downsample(mar)
    air_plot = _downsample(air)
    ax_full.plot(mar_plot.index, mar_plot.values, lw=0.6, color="#1f77b4", label="Mar")
    ax_full.plot(air_plot.index, air_plot.values, lw=0.6, color="#ff7f0e", label="Air")
    ax_full.set_title("Full time series")
    ax_full.set_xlabel("Time")
    ax_full.set_ylabel("Value")
    ax_full.legend(loc="upper left")

    # 2. Transition window
    ax_transition.plot(
        mar_transition.index, mar_transition.values, ".-", ms=2, lw=0.7,
        color="#1f77b4", label="Mar (before move)"
    )
    ax_transition.plot(
        air_transition.index, air_transition.values, ".-", ms=2, lw=0.7,
        color="#ff7f0e", label="Air (after move)"
    )
    if not mar_transition.empty and not air_transition.empty:
        ax_transition.axvline(mar_transition.index.max(), color="grey", ls="--", lw=1)
        ax_transition.axvline(air_transition.index.min(), color="grey", ls="--", lw=1)
    ax_transition.set_title(f"Transition window (±{transition_days:g} days)")
    ax_transition.set_xlabel("Time")
    ax_transition.tick_params(axis="x", rotation=20)
    ax_transition.legend(loc="best")

    # 3. Distribution (same-season subset)
    mar_values = mar_common.dropna().to_numpy(dtype=float)
    air_values = air_common.dropna().to_numpy(dtype=float)
    if circular:
        low, high, bins = 0.0, 360.0, np.linspace(0.0, 360.0, 61)
        hist_title = "Direction distribution (0–360°, same-season months)"
    else:
        low, high = _histogram_limits([mar_values, air_values])
        bins = np.linspace(low, high, 60)
        hist_title = "Distribution (same-season months, clipped 0.5–99.5%)"
    ax_hist.hist(mar_values, bins=bins, density=True, alpha=0.55,
                 color="#1f77b4", label="Mar")
    ax_hist.hist(air_values, bins=bins, density=True, alpha=0.55,
                 color="#ff7f0e", label="Air")
    ax_hist.set_title(hist_title)
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="best")

    # 4. Monthly means
    ax_monthly.plot(monthly_ref.index, monthly_ref.values, "o-", color="#1f77b4",
                    label="Mar")
    ax_monthly.plot(monthly_comp.index, monthly_comp.values, "o-", color="#ff7f0e",
                    label="Air")
    ax_monthly.set_title("Mean by calendar month")
    ax_monthly.set_xlabel("Month")
    ax_monthly.set_xticks(range(1, 13))
    ax_monthly.legend(loc="best")

    # 5. Mean diurnal cycle
    ax_diurnal.plot(diurnal_ref.index, diurnal_ref.values, "o-", color="#1f77b4",
                    label="Mar")
    ax_diurnal.plot(diurnal_comp.index, diurnal_comp.values, "o-", color="#ff7f0e",
                    label="Air")
    ax_diurnal.set_title("Mean by hour of day")
    ax_diurnal.set_xlabel("Hour")
    ax_diurnal.set_xticks(range(0, 24, 3))
    ax_diurnal.legend(loc="best")

    # 6. Boxplot (same-season subset, without outliers)
    if circular:
        ax_box.text(
            0.5, 0.5,
            "Circular variable\nlinear boxplot not meaningful",
            ha="center", va="center", transform=ax_box.transAxes, fontsize=12,
        )
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.set_title("Same-season spread")
    else:
        ax_box.boxplot(
            [mar_values, air_values],
            tick_labels=["Mar", "Air"],
            showfliers=False,
        )
        ax_box.set_title("Same-season spread")
        ax_box.set_ylabel("Value")

    fig.suptitle(f"{variable}: Mar vs Air", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / f"{variable}_comparison.png", dpi=150)
    plt.close(fig)


def _format_report(metrics: pd.DataFrame, transition_days: float) -> str:
    """Human-readable console/report summary."""
    headline_columns = [
        "variable",
        "mar_mean_common",
        "air_mean_common",
        "mean_rel_diff_pct_common",
        "circular_mean_diff_deg_common",
        "cohens_d_common",
        "cliffs_delta_common",
        "ks_stat_common",
        "monthly_corr",
        "diurnal_corr",
        "assessment",
    ]
    available_headline = [c for c in headline_columns if c in metrics.columns]

    transition_columns = [
        "variable",
        "mar_mean_transition",
        "air_mean_transition",
        "mean_rel_diff_pct_transition",
        "circular_mean_diff_deg_transition",
        "cohens_d_transition",
        "ks_stat_transition",
        "transition_gap_hours",
    ]
    available_transition = [c for c in transition_columns if c in metrics.columns]

    def _table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame.loc[:, columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4g}",
        )

    lines = [
        "=" * 110,
        "AIR vs MAR COMPARISON",
        "=" * 110,
        "The two series do not overlap in time; 'Air' values are compared against",
        "'Mar' as the reference (before the move). Positive differences mean Air",
        "is larger than Mar.",
        "",
        "Same-season comparison: calendar months present in both series with",
        "sufficient 10-minute coverage. It reduces seasonal bias but cannot",
        "remove differences caused by weather, years or the move itself.",
        "",
        "Assessment thresholds (screening aid only):",
        "  strong   : |mean rel. diff| >= 50% or |Cohen's d| >= 0.8 or KS >= 0.5",
        "  moderate : |mean rel. diff| >= 20% or |Cohen's d| >= 0.5 or KS >= 0.3",
        "",
        "Same-season metrics:",
        _table(metrics, available_headline),
        "",
        f"Transition comparison (last/first {transition_days:g} days around the move):",
        _table(metrics, available_transition),
        "",
        "Notes:",
        "  - KS p-values are omitted from the console table because they become",
        "    extremely small with this many samples; effect sizes are more useful.",
        "  - The transition window can be confounded by changing weather.",
        "  - 'dir' is circular: use circular_mean_diff_deg, not the percent",
        "    difference or Cohen's d (those are blank for 'dir').",
        "  - Full metrics and summary statistics are written next to this report.",
        "=" * 110,
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    pairs = _discover_pairs(args.data_dir, args.variables)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    summary_rows: list[dict] = []

    for variable, (mar_path, air_path) in pairs.items():
        mar = _read_series(mar_path)
        air = _read_series(air_path)

        row, summaries = _compare_variable(variable, mar, air, args)
        metric_rows.append(row)
        summary_rows.extend(summaries)

        months = _common_months(mar, air, args.min_month_coverage)
        if not args.no_plots:
            mar_transition, air_transition, _ = _transition_windows(
                mar, air, args.transition_days
            )
            _plot_variable(
                variable=variable,
                mar=mar,
                air=air,
                months=months,
                mar_transition=mar_transition,
                air_transition=air_transition,
                transition_days=args.transition_days,
                output_dir=args.output_dir,
            )

        print(f"Processed {variable}: {len(months)} common months")

    metrics = pd.DataFrame(metric_rows)
    summaries = pd.DataFrame(summary_rows)

    metrics_path = args.output_dir / "comparison_metrics.csv"
    summaries_path = args.output_dir / "summary_statistics.csv"
    report_path = args.output_dir / "report.txt"

    metrics.to_csv(metrics_path, index=False)
    summaries.to_csv(summaries_path, index=False)

    report = _format_report(metrics, args.transition_days)
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nMetrics:    {metrics_path}")
    print(f"Summaries:  {summaries_path}")
    print(f"Report:     {report_path}")
    if not args.no_plots:
        print(f"Plots:      {args.output_dir / '<variable>_comparison.png'}")


if __name__ == "__main__":
    main()
