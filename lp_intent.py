"""LP visitor intent segmentation analysis.

This script is intentionally written as a reproducible take-home project, not a
one-off notebook export. It:
  1) runs data QA and exact deduplication,
  2) builds a behavior-only pre-form intent segmentation,
  3) benchmarks the behavior model against richer and leaky alternatives,
  4) evaluates landing-page variant A vs B with raw and adjusted reads,
  5) analyzes lead quality using appointment_set,
  6) saves stakeholder-ready tables and figures.

Run from this folder:
    python lp_intent.py --input lp_sessions.csv --output-dir outputs --figure-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    test_proportions_2indep,
)

RANDOM_STATE = 42
TIER_ORDER = ["High", "Medium", "Low"]
BOUNCE_DURATION_THRESHOLD_SEC = 1.0  # near-instant sessions treated as a bot/bounce robustness check

# Main segmentation features. These are deliberately behavior-only and pre-form.
# form_started and form_field_interactions are excluded because they are too close
# to the conversion event and would make the model less useful for early scoring.
BEHAVIOR_NUMERIC_FEATURES = [
    "log_duration",
    "scroll_depth_pct",
    "log_clicks",
    "sections_viewed",
    "log_time_to_first_scroll",
]
BEHAVIOR_CATEGORICAL_FEATURES: List[str] = []

# A richer scoring model is useful as a benchmark and for media optimization, but
# not as the primary CRO segmentation because it mixes behavior with channel mix.
CONTEXT_NUMERIC_FEATURES = BEHAVIOR_NUMERIC_FEATURES + ["returning_visitor"]
CONTEXT_CATEGORICAL_FEATURES = [
    "traffic_source",
    "campaign_id",
    "device_type",
    "geo_region",
]

LEAKY_NUMERIC_FEATURES = CONTEXT_NUMERIC_FEATURES + [
    "form_started",
    "form_field_interactions",
]
LEAKY_CATEGORICAL_FEATURES = CONTEXT_CATEGORICAL_FEATURES


def safe_rate(success: float, n: float) -> float:
    return float(success / n) if n else np.nan


def wilson(success: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(success, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def load_and_prepare(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    duplicate_rows = int(raw.duplicated().sum())
    duplicate_session_ids = int(raw["session_id"].duplicated().sum())

    df = raw.drop_duplicates().copy()
    df["appointment_overall"] = (df["appointment_set"] == 1).astype(int)
    df["log_duration"] = np.log1p(df["session_duration_sec"])
    df["log_clicks"] = np.log1p(df["num_clicks"])
    df["log_time_to_first_scroll"] = np.log1p(df["time_to_first_scroll_sec"])
    df["session_date"] = df["timestamp"].dt.date.astype(str)

    qa_rows = [
        {"check": "raw_rows", "value": len(raw)},
        {"check": "rows_after_exact_dedup", "value": len(df)},
        {"check": "exact_duplicate_rows_removed", "value": duplicate_rows},
        {"check": "duplicate_session_ids_in_raw", "value": duplicate_session_ids},
        {"check": "base_lead_rate_after_dedup", "value": df["converted"].mean()},
        {"check": "appointments", "value": int(df["appointment_overall"].sum())},
        {"check": "appointment_set_missing_rate", "value": df["appointment_set"].isna().mean()},
        {"check": "scroll_depth_missing_rate", "value": df["scroll_depth_pct"].isna().mean()},
        {"check": "time_to_first_scroll_missing_rate", "value": df["time_to_first_scroll_sec"].isna().mean()},
        {
            "check": f"near_instant_bounce_sessions_lt_{BOUNCE_DURATION_THRESHOLD_SEC:g}s",
            "value": int((df["session_duration_sec"] < BOUNCE_DURATION_THRESHOLD_SEC).sum()),
        },
        {
            "check": "near_instant_bounce_share",
            "value": float((df["session_duration_sec"] < BOUNCE_DURATION_THRESHOLD_SEC).mean()),
        },
        {"check": "min_session_date", "value": df["timestamp"].min()},
        {"check": "max_session_date", "value": df["timestamp"].max()},
    ]
    return df, pd.DataFrame(qa_rows)


def make_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
                    ]
                ),
                categorical_features,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )


def out_of_fold_predictions(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target: str = "converted",
) -> Tuple[np.ndarray, Dict[str, float]]:
    features = numeric_features + categorical_features
    X = df[features]
    y = df[target].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = np.zeros(len(df))

    for train_idx, test_idx in cv.split(X, y):
        model = make_pipeline(numeric_features, categorical_features)
        model.fit(X.iloc[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "brier_score": brier_score_loss(y, scores),
        "log_loss": log_loss(y, scores),
        "mean_score": float(scores.mean()),
        "base_rate": float(y.mean()),
    }
    return scores, metrics


def model_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    configs = {
        "behavior_only_pre_form": (BEHAVIOR_NUMERIC_FEATURES, BEHAVIOR_CATEGORICAL_FEATURES),
        "behavior_plus_context": (CONTEXT_NUMERIC_FEATURES, CONTEXT_CATEGORICAL_FEATURES),
        "late_stage_leakage_check": (LEAKY_NUMERIC_FEATURES, LEAKY_CATEGORICAL_FEATURES),
    }
    rows = []
    predictions: Dict[str, np.ndarray] = {}
    for name, (num, cat) in configs.items():
        scores, metrics = out_of_fold_predictions(df, num, cat)
        predictions[name] = scores
        rows.append({"model": name, **metrics, "n_features_raw": len(num) + len(cat)})
    return pd.DataFrame(rows), predictions


def add_intent_tiers(df: pd.DataFrame, scores: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored = df.copy()
    scored["intent_score"] = scores
    low_medium_cutoff, medium_high_cutoff = np.quantile(scores, [0.50, 0.80])
    scored["intent_tier"] = np.where(
        scored["intent_score"] >= medium_high_cutoff,
        "High",
        np.where(scored["intent_score"] >= low_medium_cutoff, "Medium", "Low"),
    )
    thresholds = pd.DataFrame(
        [
            {"threshold": "low_medium_cutoff", "score": low_medium_cutoff},
            {"threshold": "medium_high_cutoff", "score": medium_high_cutoff},
        ]
    )
    return scored, thresholds


def group_summary(scored: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, g in scored.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        sessions = len(g)
        leads = int(g["converted"].sum())
        appointments = int(g["appointment_overall"].sum())
        lead_rate = safe_rate(leads, sessions)
        app_per_session = safe_rate(appointments, sessions)
        app_per_lead = safe_rate(appointments, leads)
        lead_ci_low, lead_ci_high = wilson(leads, sessions)
        app_s_ci_low, app_s_ci_high = wilson(appointments, sessions)
        app_l_ci_low, app_l_ci_high = wilson(appointments, leads)
        rows.append(
            {
                **key_dict,
                "sessions": sessions,
                "session_share": sessions / len(scored),
                "lead_rate": lead_rate,
                "lead_rate_ci95_low": lead_ci_low,
                "lead_rate_ci95_high": lead_ci_high,
                "leads": leads,
                "appointment_rate_per_lead": app_per_lead,
                "appointment_rate_per_lead_ci95_low": app_l_ci_low,
                "appointment_rate_per_lead_ci95_high": app_l_ci_high,
                "appointments": appointments,
                "appointment_rate_per_session": app_per_session,
                "appointment_rate_per_session_ci95_low": app_s_ci_low,
                "appointment_rate_per_session_ci95_high": app_s_ci_high,
            }
        )
    return pd.DataFrame(rows)


def summarize_tiers(scored: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    tier_summary = group_summary(scored, ["intent_tier"])
    tier_summary["intent_tier"] = pd.Categorical(tier_summary["intent_tier"], TIER_ORDER, ordered=True)
    tables["tier_summary"] = tier_summary.sort_values("intent_tier")

    behavior = (
        scored.groupby("intent_tier")
        .agg(
            score_min=("intent_score", "min"),
            score_median=("intent_score", "median"),
            score_max=("intent_score", "max"),
            duration_median_sec=("session_duration_sec", "median"),
            scroll_median_pct=("scroll_depth_pct", "median"),
            clicks_median=("num_clicks", "median"),
            sections_median=("sections_viewed", "median"),
            time_to_first_scroll_median_sec=("time_to_first_scroll_sec", "median"),
            returning_rate=("returning_visitor", "mean"),
            form_start_rate=("form_started", "mean"),
            form_field_interactions_median=("form_field_interactions", "median"),
        )
        .reindex(TIER_ORDER)
        .reset_index()
    )
    tables["tier_behavior"] = behavior

    for grouping, name in [
        (["landing_page_variant"], "variant_raw"),
        (["intent_tier", "landing_page_variant"], "tier_by_variant"),
        (["traffic_source"], "source_quality"),
        (["intent_tier", "traffic_source"], "tier_by_source"),
        (["intent_tier", "traffic_source", "device_type"], "tier_source_device_quality"),
    ]:
        t = group_summary(scored, grouping)
        if "intent_tier" in t.columns:
            t["intent_tier"] = pd.Categorical(t["intent_tier"], TIER_ORDER, ordered=True)
            t = t.sort_values(["intent_tier"] + [c for c in grouping if c != "intent_tier"])
        tables[name] = t.reset_index(drop=True)

    return tables


def variant_tests(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, label in [("converted", "lead_conversion"), ("appointment_overall", "appointment_per_session")]:
        g = scored.groupby("landing_page_variant")[outcome].agg(["sum", "count"])
        a_success, a_n = int(g.loc["A", "sum"]), int(g.loc["A", "count"])
        b_success, b_n = int(g.loc["B", "sum"]), int(g.loc["B", "count"])
        test = test_proportions_2indep(b_success, b_n, a_success, a_n)
        ci_low, ci_high = confint_proportions_2indep(
            b_success, b_n, a_success, a_n, method="wald"
        )
        rows.append(
            {
                "metric": label,
                "A_rate": safe_rate(a_success, a_n),
                "B_rate": safe_rate(b_success, b_n),
                "B_minus_A": safe_rate(b_success, b_n) - safe_rate(a_success, a_n),
                "B_minus_A_ci95_low": ci_low,
                "B_minus_A_ci95_high": ci_high,
                "p_value": test.pvalue,
            }
        )

    leads = scored[scored["converted"] == 1]
    g = leads.groupby("landing_page_variant")["appointment_overall"].agg(["sum", "count"])
    a_success, a_n = int(g.loc["A", "sum"]), int(g.loc["A", "count"])
    b_success, b_n = int(g.loc["B", "sum"]), int(g.loc["B", "count"])
    test = test_proportions_2indep(b_success, b_n, a_success, a_n)
    ci_low, ci_high = confint_proportions_2indep(
        b_success, b_n, a_success, a_n, method="wald"
    )
    rows.append(
        {
            "metric": "appointment_per_lead",
            "A_rate": safe_rate(a_success, a_n),
            "B_rate": safe_rate(b_success, b_n),
            "B_minus_A": safe_rate(b_success, b_n) - safe_rate(a_success, a_n),
            "B_minus_A_ci95_low": ci_low,
            "B_minus_A_ci95_high": ci_high,
            "p_value": test.pvalue,
        }
    )
    return pd.DataFrame(rows)


def adjusted_variant_read(scored: pd.DataFrame) -> pd.DataFrame:
    """Adjusted A/B read using pre-treatment / allocation controls.

    I do not control for session behavior here because duration, scroll, and clicks
    may be downstream effects of the page variant itself.
    """
    rows = []
    controls = "C(campaign_id) + C(device_type) + C(geo_region) + returning_visitor"
    for outcome, label in [("converted", "lead_conversion"), ("appointment_overall", "appointment_per_session")]:
        model = smf.glm(f"{outcome} ~ C(landing_page_variant) + {controls}", data=scored, family=sm.families.Binomial()).fit()
        tmp_a = scored.copy()
        tmp_b = scored.copy()
        tmp_a["landing_page_variant"] = "A"
        tmp_b["landing_page_variant"] = "B"
        pred_a = float(model.predict(tmp_a).mean())
        pred_b = float(model.predict(tmp_b).mean())
        coef = model.params["C(landing_page_variant)[T.B]"]
        se = model.bse["C(landing_page_variant)[T.B]"]
        rows.append(
            {
                "metric": label,
                "adjusted_A_rate": pred_a,
                "adjusted_B_rate": pred_b,
                "adjusted_B_minus_A": pred_b - pred_a,
                "odds_ratio_B_vs_A": float(np.exp(coef)),
                "or_ci95_low": float(np.exp(coef - 1.96 * se)),
                "or_ci95_high": float(np.exp(coef + 1.96 * se)),
                "p_value": float(model.pvalues["C(landing_page_variant)[T.B]"]),
            }
        )
    return pd.DataFrame(rows)


def variant_mix_tables(scored: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables = {}
    for col in ["device_type", "traffic_source", "campaign_id", "geo_region"]:
        counts = pd.crosstab(scored["landing_page_variant"], scored[col])
        shares = pd.crosstab(scored["landing_page_variant"], scored[col], normalize="index")
        counts = counts.reset_index().melt("landing_page_variant", var_name=col, value_name="sessions")
        shares = shares.reset_index().melt("landing_page_variant", var_name=col, value_name="share")
        tables[f"variant_mix_{col}"] = counts.merge(shares, on=["landing_page_variant", col])

    tables["variant_by_device"] = group_summary(scored, ["device_type", "landing_page_variant"])
    tables["variant_by_source"] = group_summary(scored, ["traffic_source", "landing_page_variant"])
    tables["variant_by_campaign"] = group_summary(scored, ["campaign_id", "landing_page_variant"])
    return tables


def decile_tables(scored: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tmp = scored.copy()
    # Decile 10 is the highest predicted intent decile.
    tmp["score_decile"] = pd.qcut(tmp["intent_score"].rank(method="first"), 10, labels=False) + 1
    tmp["score_decile"] = 11 - tmp["score_decile"]
    decile = (
        tmp.groupby("score_decile")
        .agg(
            sessions=("session_id", "size"),
            mean_score=("intent_score", "mean"),
            lead_rate=("converted", "mean"),
            leads=("converted", "sum"),
            appointment_rate_per_session=("appointment_overall", "mean"),
            appointments=("appointment_overall", "sum"),
        )
        .sort_index()
        .reset_index()
    )
    decile["lift_vs_base_lead_rate"] = decile["lead_rate"] / scored["converted"].mean()
    return {"lift_by_decile": decile, "calibration_by_decile": decile[["score_decile", "sessions", "mean_score", "lead_rate"]]}


def behavior_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    model = make_pipeline(BEHAVIOR_NUMERIC_FEATURES, BEHAVIOR_CATEGORICAL_FEATURES)
    model.fit(df[BEHAVIOR_NUMERIC_FEATURES], df["converted"])
    coefs = model.named_steps["model"].coef_.ravel()
    rows = []
    for feature, coef in zip(BEHAVIOR_NUMERIC_FEATURES, coefs):
        rows.append({"feature": feature, "standardized_log_odds_coef": coef, "odds_ratio_per_1sd": np.exp(coef)})
    return pd.DataFrame(rows).sort_values("standardized_log_odds_coef", ascending=False)


def robustness_excluding_bounces(scored: pd.DataFrame) -> pd.DataFrame:
    """Re-run the tier summary excluding near-instant bounce/bot-like sessions.

    These sessions should mechanically fall into the Low tier already, so this is a
    sanity check that they are not silently driving the tier story, not a data cleaning
    step applied to the main analysis.
    """
    clean = scored[scored["session_duration_sec"] >= BOUNCE_DURATION_THRESHOLD_SEC]
    out = group_summary(clean, ["intent_tier"])
    out["intent_tier"] = pd.Categorical(out["intent_tier"], TIER_ORDER, ordered=True)
    return out.sort_values("intent_tier").reset_index(drop=True)


def variant_temporal_stability(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether variant B looks like a concurrent split or a phased rollout.

    "B is a recent redesign" invites the question of whether B's traffic share (and thus
    any seasonality/campaign effects) shifted over the collection window. If it did, the
    A/B comparison would be confounded by time, not just by device mix. This fits a simple
    day-index vs. is-B logistic trend model and reports weekly B share for a visual check.
    """
    tmp = scored.copy()
    tmp["session_date"] = pd.to_datetime(tmp["session_date"])
    tmp["week"] = tmp["session_date"].dt.to_period("W").astype(str)
    weekly = (
        tmp.groupby("week")
        .agg(sessions=("session_id", "size"), b_share=("landing_page_variant", lambda s: (s == "B").mean()))
        .reset_index()
    )

    tmp["day_index"] = (tmp["session_date"] - tmp["session_date"].min()).dt.days
    tmp["is_b"] = (tmp["landing_page_variant"] == "B").astype(int)
    trend_model = smf.logit("is_b ~ day_index", data=tmp).fit(disp=0)
    trend = pd.DataFrame(
        [
            {
                "check": "variant_B_share_trend_over_time",
                "coef_per_day": trend_model.params["day_index"],
                "p_value": trend_model.pvalues["day_index"],
                "overall_B_share": float(tmp["is_b"].mean()),
                "min_weekly_B_share": float(weekly["b_share"].min()),
                "max_weekly_B_share": float(weekly["b_share"].max()),
            }
        ]
    )
    return weekly, trend


def add_multiplicity_correction(variant_tests_df: pd.DataFrame) -> pd.DataFrame:
    """Holm-Bonferroni correction across the raw A/B tests run on the same sample.

    Three outcomes (lead conversion, appointment/session, appointment/lead) are tested on
    overlapping data. This adds adjusted p-values so a reviewer can see the raw read holds
    up (or doesn't) after correcting for multiple looks, without changing the point estimates.
    """
    out = variant_tests_df.copy()
    reject, p_adj, _, _ = multipletests(out["p_value"], alpha=0.05, method="holm")
    out["p_value_holm_adjusted"] = p_adj
    out["significant_after_holm_alpha_0.05"] = reject
    return out


class RealtimeIntentScorer:
    """Scores a session in progress using the same fitted behavior-only pipeline.

    lp_sessions.csv is one row per *completed* session, not an event log, so there is no
    true sub-session ground truth in this dataset to validate a real-time model against.
    This class and `simulate_realtime_trajectories` below exist to demonstrate how the
    pre-form design described in the write-up would actually be implemented, not to claim
    validated real-time accuracy.
    """

    def __init__(self, fitted_pipeline: Pipeline, low_medium_cutoff: float, medium_high_cutoff: float):
        self.pipeline = fitted_pipeline
        self.low_medium_cutoff = low_medium_cutoff
        self.medium_high_cutoff = medium_high_cutoff

    def score_partial(
        self,
        elapsed_sec: float,
        scroll_depth_pct_so_far: float,
        clicks_so_far: int,
        sections_viewed_so_far: int,
        time_to_first_scroll_sec: float,
    ) -> Dict[str, float]:
        row = pd.DataFrame(
            [
                {
                    "log_duration": np.log1p(max(elapsed_sec, 0)),
                    "scroll_depth_pct": scroll_depth_pct_so_far,
                    "log_clicks": np.log1p(max(clicks_so_far, 0)),
                    "sections_viewed": sections_viewed_so_far,
                    "log_time_to_first_scroll": np.log1p(max(time_to_first_scroll_sec, 0)),
                }
            ]
        )
        score = float(self.pipeline.predict_proba(row)[:, 1][0])
        if score >= self.medium_high_cutoff:
            tier = "High"
        elif score >= self.low_medium_cutoff:
            tier = "Medium"
        else:
            tier = "Low"
        return {"elapsed_sec": elapsed_sec, "score": score, "live_tier": tier}


def fit_final_behavior_pipeline(df: pd.DataFrame) -> Pipeline:
    pipeline = make_pipeline(BEHAVIOR_NUMERIC_FEATURES, BEHAVIOR_CATEGORICAL_FEATURES)
    pipeline.fit(df[BEHAVIOR_NUMERIC_FEATURES], df["converted"])
    return pipeline


def simulate_realtime_trajectories(
    scored: pd.DataFrame,
    scorer: RealtimeIntentScorer,
    n_per_group: int = 3,
    checkpoints: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Simulate progressive in-session scoring for a handful of real completed sessions.

    Behavior is assumed to accumulate roughly proportionally to elapsed time within a
    session. Real sessions burst and scroll unevenly, so this is a simplification used to
    show the *shape* of how a live score would evolve and how early a high-intent signal
    could plausibly fire relative to actual form completion, not a validated event model.
    """
    rng = np.random.RandomState(random_state)
    groups = []
    for outcome_label, subset in [
        ("became_lead", scored[scored["converted"] == 1]),
        ("did_not_convert", scored[scored["converted"] == 0]),
    ]:
        picked = subset.sample(min(n_per_group, len(subset)), random_state=rng)
        groups.append(picked.assign(outcome_label=outcome_label))
    sample = pd.concat(groups)

    rows = []
    for _, sess in sample.iterrows():
        scroll_final = sess["scroll_depth_pct"] if pd.notna(sess["scroll_depth_pct"]) else 0.0
        ttfs = sess["time_to_first_scroll_sec"] if pd.notna(sess["time_to_first_scroll_sec"]) else 0.0
        for frac in checkpoints:
            partial = scorer.score_partial(
                elapsed_sec=sess["session_duration_sec"] * frac,
                scroll_depth_pct_so_far=min(scroll_final * frac, 100),
                clicks_so_far=int(round(sess["num_clicks"] * frac)),
                sections_viewed_so_far=int(round(sess["sections_viewed"] * frac)),
                # first-scroll timing is known as soon as it happens, not interpolated
                time_to_first_scroll_sec=ttfs if sess["session_duration_sec"] * frac >= ttfs else frac * ttfs,
            )
            rows.append(
                {
                    "session_id": sess["session_id"],
                    "outcome_label": sess["outcome_label"],
                    "final_intent_tier": sess["intent_tier"],
                    "checkpoint_frac_of_session": frac,
                    **partial,
                }
            )
    return pd.DataFrame(rows)


def make_figures(tables: Dict[str, pd.DataFrame], figure_dir: str | Path) -> None:
    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)

    tier = tables["tier_summary"].copy()
    tier["intent_tier"] = pd.Categorical(tier["intent_tier"], TIER_ORDER, ordered=True)
    tier = tier.sort_values("intent_tier")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = tier["lead_rate"] * 100
    ax.bar(tier["intent_tier"].astype(str), vals)
    ax.set_title("Lead conversion by behavior-based intent tier")
    ax.set_ylabel("Lead conversion rate (%)")
    ax.set_xlabel("Intent tier")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "conversion_by_intent_tier.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    vals = tier["appointment_rate_per_session"] * 100
    ax.bar(tier["intent_tier"].astype(str), vals)
    ax.set_title("Appointment yield by behavior-based intent tier")
    ax.set_ylabel("Appointments per session (%)")
    ax.set_xlabel("Intent tier")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "appointment_yield_by_intent_tier.png", dpi=180)
    plt.close(fig)

    dec = tables["lift_by_decile"].sort_values("score_decile")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(dec["score_decile"], dec["lead_rate"] * 100, marker="o")
    ax.set_title("Lead rate by predicted intent decile")
    ax.set_xlabel("Predicted intent decile, 1 = highest")
    ax.set_ylabel("Lead conversion rate (%)")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out / "lead_rate_by_decile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(dec["mean_score"] * 100, dec["lead_rate"] * 100)
    ax.plot([0, max(dec["mean_score"].max(), dec["lead_rate"].max()) * 100], [0, max(dec["mean_score"].max(), dec["lead_rate"].max()) * 100])
    ax.set_title("Calibration by intent decile")
    ax.set_xlabel("Mean predicted lead probability (%)")
    ax.set_ylabel("Observed lead conversion rate (%)")
    fig.tight_layout()
    fig.savefig(out / "calibration_by_decile.png", dpi=180)
    plt.close(fig)

    weekly = tables["variant_b_weekly_share"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(range(len(weekly)), weekly["b_share"] * 100, marker="o")
    ax.axhline(weekly["b_share"].mean() * 100, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(weekly)))
    ax.set_xticklabels(weekly["week"], rotation=45, ha="right")
    ax.set_title("Variant B share of traffic by week (checking for a phased rollout)")
    ax.set_ylabel("Share of sessions on variant B (%)")
    ax.set_xlabel("Week")
    fig.tight_layout()
    fig.savefig(out / "variant_b_share_over_time.png", dpi=180)
    plt.close(fig)

    traj = tables["realtime_scoring_demo"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"became_lead": "tab:green", "did_not_convert": "tab:red"}
    seen_labels = set()
    for sid, g in traj.groupby("session_id"):
        g = g.sort_values("checkpoint_frac_of_session")
        label = g["outcome_label"].iloc[0]
        plot_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.plot(g["checkpoint_frac_of_session"] * 100, g["score"] * 100, marker="o", color=colors[label], alpha=0.7, label=plot_label)
    ax.axhline(tables["intent_thresholds"].set_index("threshold").loc["medium_high_cutoff", "score"] * 100, linestyle="--", linewidth=1, color="gray")
    ax.set_title("Simulated real-time intent score as a session unfolds")
    ax.set_xlabel("Elapsed share of eventual session (%)")
    ax.set_ylabel("Live predicted lead probability (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "realtime_scoring_trajectories.png", dpi=180)
    plt.close(fig)

    vt = tables["variant_tests"].set_index("metric")
    va = tables["variant_adjusted"].set_index("metric")
    plot_rows = []
    for metric, label in [("lead_conversion", "Lead conversion"), ("appointment_per_session", "Appointment/session")]:
        plot_rows.append({"metric": label, "read": "Raw A", "rate": vt.loc[metric, "A_rate"]})
        plot_rows.append({"metric": label, "read": "Raw B", "rate": vt.loc[metric, "B_rate"]})
        plot_rows.append({"metric": label, "read": "Adjusted A", "rate": va.loc[metric, "adjusted_A_rate"]})
        plot_rows.append({"metric": label, "read": "Adjusted B", "rate": va.loc[metric, "adjusted_B_rate"]})
    plot = pd.DataFrame(plot_rows)
    # Simple grouped chart without setting custom colors.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = plot["metric"].unique().tolist()
    reads = plot["read"].unique().tolist()
    width = 0.18
    x = np.arange(len(labels))
    for i, r in enumerate(reads):
        vals = [plot[(plot["metric"] == m) & (plot["read"] == r)]["rate"].iloc[0] * 100 for m in labels]
        ax.bar(x + (i - 1.5) * width, vals, width, label=r)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Variant A vs B: raw vs adjusted read")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "variant_raw_vs_adjusted.png", dpi=180)
    plt.close(fig)


def save_all(scored: pd.DataFrame, tables: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "intent_scored_sessions.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="lp_sessions.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    df, qa = load_and_prepare(args.input)
    comparison, predictions = model_comparison(df)
    scored, thresholds = add_intent_tiers(df, predictions["behavior_only_pre_form"])

    tables: Dict[str, pd.DataFrame] = {
        "data_quality_summary": qa,
        "model_comparison": comparison,
        "intent_thresholds": thresholds,
        "behavior_model_coefficients": behavior_coefficients(df),
    }
    tables.update(summarize_tiers(scored))
    tables["tier_summary_excluding_bounces"] = robustness_excluding_bounces(scored)
    tables["variant_tests"] = add_multiplicity_correction(variant_tests(scored))
    tables["variant_adjusted"] = adjusted_variant_read(scored)
    tables.update(variant_mix_tables(scored))
    tables.update(decile_tables(scored))

    weekly, trend = variant_temporal_stability(scored)
    tables["variant_b_weekly_share"] = weekly
    tables["variant_b_temporal_trend_test"] = trend

    final_pipeline = fit_final_behavior_pipeline(df)
    thresh = thresholds.set_index("threshold")["score"]
    scorer = RealtimeIntentScorer(
        final_pipeline,
        low_medium_cutoff=thresh["low_medium_cutoff"],
        medium_high_cutoff=thresh["medium_high_cutoff"],
    )
    tables["realtime_scoring_demo"] = simulate_realtime_trajectories(scored, scorer)

    save_all(scored, tables, args.output_dir)
    make_figures(tables, args.figure_dir)

    print("Rows after exact deduplication:", len(df))
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nTier summary:")
    print(tables["tier_summary"].to_string(index=False))
    print("\nTier summary excluding near-instant bounces:")
    print(tables["tier_summary_excluding_bounces"].to_string(index=False))
    print("\nVariant tests (with Holm correction):")
    print(tables["variant_tests"].to_string(index=False))
    print("\nAdjusted variant read:")
    print(tables["variant_adjusted"].to_string(index=False))
    print("\nVariant B temporal stability check:")
    print(tables["variant_b_temporal_trend_test"].to_string(index=False))


if __name__ == "__main__":
    main()
