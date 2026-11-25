"""Small static figures and a report grounded entirely in saved study aggregates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def table(frame: pl.DataFrame, columns: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.select(list(columns)).iter_rows():
        values = [
            "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v) for v in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def figures(output, summary, segments, costs, reference, horizon):
    folder = output / "charts"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for group in summary.partition_by("pair"):
        ax.plot(group["horizon"], group["full_resolved_pct"], marker="o", label=group["pair"][0])
    ax.set(
        xlabel="Holding limit (minutes)",
        ylabel="First price barrier resolved (%)",
        title="Common cohort with the full maximum window observed",
        ylim=(0, 100),
    )
    ax.legend()
    save(fig, "resolution.png")

    names = list(dict.fromkeys([reference, "wide" if "wide" in summary["pair"] else reference]))
    fig, axes = plt.subplots(1, len(names), figsize=(9, 4), squeeze=False)
    outcome_fields = ["tp_first_pct", "sl_first_pct", "expire_pct", "ambiguous_pct", "unknown_pct"]
    colors = ["#269b69", "#cf5858", "#5998cc", "#e4ad43", "#999999"]
    for ax, name in zip(axes[0], names, strict=True):
        rows = summary.filter(pl.col("pair") == name)
        bottom = np.zeros(rows.height)
        for column, color in zip(outcome_fields, colors, strict=True):
            ax.bar(
                rows["horizon"].cast(pl.String),
                rows[column],
                bottom=bottom,
                label=column.removesuffix("_pct"),
                color=color,
            )
            bottom += rows[column].to_numpy()
        ax.set(
            title=name, xlabel="Holding limit (minutes)", ylabel="All candidates (%)", ylim=(0, 100)
        )
    axes[0, -1].legend(fontsize=7)
    save(fig, "outcomes.png")

    hist = pl.read_csv(output / "expiration_histogram.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in dict.fromkeys([reference, "tight", "wide"]):
        rows = hist.filter((pl.col("pair") == name) & (pl.col("horizon") == horizon))
        if rows.height:
            ax.step(
                rows["bin_left_bps"],
                rows["count"] / rows["count"].sum() * 100,
                where="post",
                label=name,
            )
    ax.set(
        xlabel="Expiration return (bps; 5 bps bins)",
        ylabel="Expirations (%)",
        title=f"Conditional expiration distributions at {horizon} minutes",
    )
    ax.legend()
    save(fig, "expiration_returns.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    policies = summary.select("pair", "horizon").rows()
    cost_grid = sorted(costs["cost_bps"].unique())
    matrix = np.array(
        [
            [
                costs.filter(
                    (pl.col("pair") == name) & (pl.col("horizon") == h) & (pl.col("cost_bps") == c)
                )["conservative_net_mean_bps"].item()
                for c in cost_grid
            ]
            for name, h in policies
        ],
        dtype=float,
    )
    limit = max(float(np.nanmax(np.abs(matrix))), 1) if np.isfinite(matrix).any() else 1
    display = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(cost_grid)), [str(c) for c in cost_grid])
    ax.set_yticks(range(len(policies)), [f"{p} / {h}m" for p, h in policies])
    ax.set(
        xlabel="Illustrative round-trip cost (bps)",
        title="Mean candidate return, conservative ambiguity",
    )
    fig.colorbar(display, ax=ax, label="Net bps / valid hypothetical entry")
    save(fig, "cost_sensitivity.png")

    rows = segments.filter(
        (pl.col("pair") == reference)
        & (pl.col("horizon") == horizon)
        & (pl.col("dimension") == "volatility")
    ).sort("bucket")
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(rows.height)
    for column, color in zip(outcome_fields, colors, strict=True):
        ax.bar(
            rows["bucket"],
            rows[column],
            bottom=bottom,
            label=column.removesuffix("_pct"),
            color=color,
        )
        bottom += rows[column].to_numpy()
    ax.set(
        title=f"Prior volatility terciles: {reference} / {horizon}m",
        xlabel="Prior daily log-return volatility",
        ylabel="All candidates (%)",
        ylim=(0, 100),
    )
    ax.legend(fontsize=7)
    save(fig, "volatility.png")


def write_report(output: Path, config, quality):
    s = pl.read_csv(output / "summary.csv")
    segments = pl.read_csv(output / "segment_summary.csv")
    fixed = pl.read_csv(output / "fixed_horizon_summary.csv")
    correlations = pl.read_csv(output / "fixed_horizon_correlations.csv")
    costs = pl.read_csv(output / "cost_sensitivity.csv")
    names = s["pair"].unique().to_list()
    reference = "original" if "original" in names else config.research.pairs[0].name
    h = min(config.research.horizons, key=lambda value: abs(value - 60))
    baseline = s.filter((pl.col("pair") == reference) & (pl.col("horizon") == h)).row(0, named=True)
    horizons = s.filter(pl.col("pair") == reference).sort("horizon")
    vol = segments.filter(
        (pl.col("pair") == reference)
        & (pl.col("horizon") == h)
        & (pl.col("dimension") == "volatility")
        & (pl.col("bucket") != "unknown")
    )
    expiry_spread = (
        vol["expire_pct"].max() - vol["expire_pct"].min() if not vol.is_empty() else None
    )
    unknown = baseline["unknown_pct"]
    q = quality
    figures(output, s, segments, costs, reference, h)
    f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    candidate_count = q["candidate_entries"]
    alternate = max(
        (
            r
            for r in horizons.iter_rows(named=True)
            if r["horizon"] != h
            and (r["horizon"] in [30, 90, 120] or len(config.research.horizons) < 3)
        ),
        key=lambda r: abs(r["expire_pct"] - baseline["expire_pct"]),
        default=baseline,
    )
    third = (
        "wide"
        if "wide" in names and reference != "wide"
        else next((name for name in names if name != reference), reference)
    )
    third_row = s.filter((pl.col("pair") == third) & (pl.col("horizon") == h)).row(0, named=True)
    decisions = sorted(
        {(b["first_decision_et"][11:16], b["last_decision_et"][11:16]) for b in q["batches"]}
    )
    ambiguity_spread = (
        f(baseline["optimistic_mean_bps"] - baseline["gross_bps_mean"])
        if baseline["valid"]
        else "n/a"
    )
    known_expiration_pct = (
        100 * baseline["expire_count"] / baseline["valid"] if baseline["valid"] else None
    )
    ambiguity_assessment = (
        "At this observed frequency and sensitivity, minute OHLC is workable for initial "
        "barrier-label research with retained ambiguity masks. Missing-data exclusions "
        "remain a separate and potentially larger source of selection bias."
        if baseline["valid"]
        and baseline["ambiguous_pct"] < 1
        and baseline["optimistic_mean_bps"] - baseline["gross_bps_mean"] < 1
        else "Ambiguity is material enough to require policy-specific sensitivity review "
        "before adopting these minute-bar labels."
    )
    positive_costs = [c for c in config.research.costs_bps if c > 0]
    cost_assessment = (
        f"At the smallest positive illustrative cost, {min(positive_costs):g} bps, "
        f"the baseline net mean is {f(baseline['gross_bps_mean'] - min(positive_costs))} bps "
        "per valid candidate. This arithmetic describes sampled hypothetical entries."
        if positive_costs and baseline["valid"]
        else "Cost sensitivities use the configured round-trip assumptions."
    )
    text = [
        "# Exploratory label study",
        "## 1. Executive summary",
        f"This run sampled **{q['sessions']} sessions**, **{q['minute_rows']:,} regular-session "
        f"bars** "
        f"and **{candidate_count:,} candidate long entries** before 2024. For {reference} / {h}m, "
        f"expiration is **{f(baseline['expire_pct'])}%**, ambiguous outcomes "
        f"**{f(baseline['ambiguous_pct'])}%**, and UNKNOWN **{f(unknown)}%** of candidates. "
        f"The mean conditional expiration return is {f(baseline['expiration_bps_mean'])} bps "
        f"with standard deviation {f(baseline['expiration_bps_std'])} bps. "
        f"The primary descriptive gross mean is {f(baseline['gross_bps_mean'])} bps per valid "
        f"entry.",
        f"Expiration represents {f(known_expiration_pct)}% of known outcomes. "
        f"UNKNOWN windows ({f(unknown)}% of candidates) require explicit coverage review "
        "before training; dropping them can change the composition of the learning sample.",
        "These overlapping hypothetical entries are candidate-level descriptions, not a "
        "capital-constrained strategy. They establish neither predictability nor achievable "
        "profitability. Class balance and favorable unconditional returns are not "
        "target-selection objectives.",
        "## 2. Configuration and coverage",
        f"Dates [{config.research.start}, {config.research.end}); seed {config.research.seed}; "
        f"{config.research.sessions_per_month} random valid session(s) per month; top "
        f"{config.research.universe_size} prior-liquidity symbols per session. "
        f"Decision spacing {config.research.decision_minutes}m; execution delay "
        f"{config.research.execution_delay}m; horizons {config.research.horizons}. "
        f"Session cap: {config.research.max_sessions or 'none'}. "
        "Exact dates and UTC opens/closes are in sampled_sessions.csv; all settings are in "
        "config.yaml.",
        f"Supporting daily feature calculation used {q['supporting_daily_observations']:,} "
        f"deduplicated valid close/volume observations for {q['daily_symbols']:,} symbols, "
        f"from {q['first_daily_date']} through {q['last_daily_date']}. "
        "Invalid/nonpositive daily closes or invalid dollar volumes are omitted from prior "
        "feature histories. Selected feature dates are strictly earlier than each decision "
        "session.",
        f"Common-cohort ET decision ranges (first, last): {decisions}. All policies use the same "
        f"scheduled entry population with room for the longest ({max(config.research.horizons)}m) "
        "window. This excludes later-session opportunities; early closes shorten the grid. "
        "No overnight positions are permitted. Eligibility never depends on subsequent bar "
        "availability.",
        "The XNYS exchange calendar supplies regular opens, holidays, DST, and early closes. "
        "Stored UTC timestamps identify minute starts. A 10:00 ET decision uses information "
        "through 09:59; at default delay, entry is the 10:01 open and the entry bar is evaluated. "
        "At 60 minutes, expiration is 11:01 using the 11:00 close. "
        "See [Alpaca timestamp conventions](https://docs.alpaca.markets/us/docs/market-data-faq) "
        "and [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars).",
        "## 3. Data quality, eligibility, and limitations",
        f"Manifest coverage was complete for {q['manifest_covered_symbols']:,}/"
        f"{q['universe_symbols']:,} sampled symbol-sessions. Expected symbol-minutes: "
        f"{q['expected_symbol_minutes']:,}; missing: {q['missing_minutes']:,}; duplicate extra "
        f"rows: "
        f"{q['duplicate_rows']:,}; invalid/nonpositive/nonfinite OHLC rows: "
        f"{q['invalid_ohlc_rows']:,}; "
        f"off-grid timestamps: {q['unaligned_rows']:,}; empty symbol-sessions: "
        f"{q['empty_symbols']:,}. "
        f"Consecutive-minute open/previous-close discontinuities above 20%: "
        f"{q['extreme_discontinuities']:,}. These flags are reported, not filtered using future "
        f"returns. "
        f"Prior-feature date violations: {q['prior_feature_date_violations']}; sampled prior "
        f"daily volatility values above 20%: {q['prior_volatility_over_20pct']}.",
        "A missing, invalid, duplicate, or unaligned minute at or before the first determinable "
        "exit makes that policy UNKNOWN. Missing minutes after a resolved exit do not invalidate "
        "it. Fixed-horizon targets require the entire window. No OHLC is forward-filled. "
        "Missing intervals may mean no eligible trades, halts, or vendor coverage gaps; "
        "completed downloads cannot distinguish these causes. UNKNOWN rates may vary by horizon "
        "and threshold, so comparisons also expose a fully observed common cohort.",
        "The existing liquidity rolling values use prior observations, but the stored daily "
        "ranking population requires a current-day bar. Research therefore recomputes top ranks "
        "from the most recent 20 *prior* daily observations over the broad daily population, "
        f"requiring a last observation within {config.research.max_staleness_days} calendar days. "
        "Same-day existence and volume never gate eligibility. Selected symbols and feature "
        "dates are persisted in selected_universe.parquet. This is a research workaround, "
        "not an ingestion redesign. An unavailable symbol's minute request causes a clear error.",
        "Prices are raw SIP and tickers are unmapped (asof=-). Asset snapshots are not a complete "
        "historical security master; survivorship, renamed/delisted tickers, stale eligibility, "
        "and ticker reuse remain limitations. Splits/dividends can distort prior close-return "
        "volatility and dollar-volume ranks; within-session discontinuity checks cannot certify "
        "corporate-action cleanliness. Vintage bars contain vendor revisions unavailable in "
        "real time. Feed trade-condition filtering and OHLC prices do not describe quotes, "
        "spread, queue priority, or actual fills. The universe includes ETFs and other instruments "
        "present in daily data, not exclusively common stocks.",
        "## 4. Policy comparison and duration",
        "Raw outcome percentages below divide by all candidates. Valid = all non-UNKNOWN, "
        "including AMBIGUOUS. Conservative TP/SL percentages in summary.csv divide by valid; "
        "ambiguity joins SL conservatively and TP optimistically. Excluded metrics remove "
        "ambiguity as well as UNKNOWN. Each row uses the same scheduled cohort.",
        table(
            s,
            {
                "pair": "Pair",
                "horizon": "Minutes",
                "tp_first_pct": "TP %",
                "sl_first_pct": "SL %",
                "expire_pct": "Expire %",
                "ambiguous_pct": "Ambiguous %",
                "unknown_pct": "Unknown %",
                "holding_minutes_mean": "Mean hold",
                "holding_minutes_median": "Median hold",
            },
        ),
        "![Cumulative first resolution](charts/resolution.png)",
        "The resolution plot uses entries valid through the longest horizon, so its cohort is "
        "constant and cumulative barrier resolution must be nondecreasing. This invariant was "
        "checked for every barrier pair. Intrabar holding times use minute end; opening exits "
        "use minute start. Actual intraminute holding times are not observable. Exact holding "
        "percentiles and outcome-specific holding means/medians/p95 are in summary.csv.",
        "![Raw outcome distributions](charts/outcomes.png)",
        f"**Does {h}m look reasonable?** It is a useful initial baseline with "
        f"{f(baseline['expire_pct'])}% expiry and {f(baseline['full_resolved_pct'])}% barrier "
        "resolution in the fully observed cohort. The table quantifies how 30, 90, and 120 "
        "minutes change the label problem; no claim of an optimal expiration follows. "
        f"The largest sampled expiry difference from this baseline is at {alternate['horizon']}m "
        f"({f(alternate['expire_pct'] - baseline['expire_pct'])} percentage points).",
        "## 5. Expiration frequency and returns",
        table(
            horizons,
            {
                "horizon": "Minutes",
                "expire_pct": "Expire %",
                "expiration_bps_mean": "Mean bps",
                "expiration_bps_median": "Median",
                "expiration_bps_std": "SD",
                "expiration_bps_p5": "p5",
                "expiration_bps_p25": "p25",
                "expiration_bps_p75": "p75",
                "expiration_bps_p95": "p95",
                "expiration_bps_positive_pct": "Positive %",
                "expiration_bps_negative_pct": "Negative %",
                "expiration_contribution_valid_bps": "Contribution bps/valid",
            },
        ),
        "![Expiration return distributions](charts/expiration_returns.png)",
        f"For {reference}/{h}m, replacing expiration returns with zero changes the valid-entry "
        f"gross mean by {f(baseline['expiration_contribution_valid_bps'])} bps. "
        f"The known expiration component per *all* candidates is "
        f"{f(baseline['expiration_contribution_candidate_bps'])} bps. "
        "These are conditional expiration sums divided by the stated unconditional cohort, "
        "not conditional means. A nonzero spread and material expiration frequency justify "
        "testing a conditional expiration-return regressor against a zero-return baseline; "
        "they do not establish that such returns are predictable or that a regressor adds value.",
        "## 6. Ambiguity and execution assumptions",
        f"Across policies, raw ambiguity ranges from {f(s['ambiguous_pct'].min())}% to "
        f"{f(s['ambiguous_pct'].max())}% of candidates. At the baseline, optimistic minus "
        f"conservative mean is "
        f"{ambiguity_spread} bps; "
        f"excluded mean is {f(baseline['excluded_mean_bps'])} bps. "
        "Interpret practicality using both these frequencies and economic spreads, especially "
        "for the tighter barriers and high-volatility stocks. Raw ambiguity evidence remains "
        "in compact first-touch checkpoints; no candle-direction ordering is assumed.",
        ambiguity_assessment,
        "Each bar's open is evaluated before its high/low. A stop gap fills at the adverse "
        "opening price; a favorable TP gap fills at the threshold conservatively. When the "
        "open is between barriers and both extremes touch, ordering is ambiguous. Additional "
        "costs are subtracted once per completed hypothetical trade; adverse gap impact is "
        "already embedded in gross returns and is not charged a second time.",
        "## 7. Hypothetical economics and cost sensitivity",
        table(
            s,
            {
                "pair": "Pair",
                "horizon": "Minutes",
                "gross_bps_mean": "Gross mean bps",
                "gross_bps_median": "Median",
                "gross_bps_std": "SD",
                "gross_bps_positive_pct": "Positive %",
                "optimistic_mean_bps": "Optimistic mean",
                "excluded_mean_bps": "Excluded mean",
                "mean_bps_per_holding_minute": "bps/min",
                "break_even_cost_bps": "Break-even bps",
            },
        ),
        "![Illustrative cost sensitivity](charts/cost_sensitivity.png)",
        cost_assessment,
        "Break-even round-trip bps equals the conservative valid-entry gross mean; a negative "
        "number means losses before positive costs. The time proxy is sum of gross bps / sum "
        "of approximate holding minutes, not an achievable turnover or capital return. "
        "Cost CSV includes conservative, optimistic, and excluded sensitivities for every "
        "configured cost. Returns divide by known exits; UNKNOWN returns are not imputed. "
        "known_return_sum_per_candidate_bps is a partial known sum/all candidates, not an "
        "estimate of missing outcomes. No annualization or portfolio Sharpe is calculated.",
        "## 8. Limited segmentation",
        "Each dimension is analyzed separately. Liquidity and volatility buckets are prior "
        "daily cross-sectional terciles within the selected universe. Volatility is sample "
        "standard deviation of the latest 20 observed daily log close-to-close returns, "
        "strictly prior to the sampled session (unannualized, raw prices). Unknown history "
        "gets its own bucket. Time buckets use decision time in New York.",
        table(
            segments.filter((pl.col("pair") == reference) & (pl.col("horizon") == h)),
            {
                "dimension": "Dimension",
                "bucket": "Bucket",
                "candidates": "Candidates",
                "expire_pct": "Expire %",
                "ambiguous_pct": "Ambiguous %",
                "unknown_pct": "Unknown %",
                "expiration_bps_std": "Expiry SD bps",
                "gross_bps_mean": "Gross bps",
            },
        ),
        "![Outcome distribution by prior volatility](charts/volatility.png)",
        f"Baseline expiry differs by {f(expiry_spread)} percentage points across known "
        "prior-volatility terciles; unknown-history observations are excluded from this spread. "
        "Larger differences mean fixed barriers create different label tasks across "
        "stocks, making volatility-scaled barriers a sensible later sensitivity experiment. "
        "They are not implemented or optimized here. Year and time-of-day differences above "
        "describe sampled sessions; a single session per month cannot establish regime stability.",
        "## 9. Fixed-horizon regression targets",
        table(
            fixed,
            {
                "horizon": "Minutes",
                "valid": "Valid",
                "return_bps_mean": "Mean bps",
                "return_bps_median": "Median",
                "return_bps_std": "SD",
                "return_bps_p5": "p5",
                "return_bps_p25": "p25",
                "return_bps_p75": "p75",
                "return_bps_p95": "p95",
                "return_bps_positive_pct": "Positive %",
            },
        ),
        table(
            correlations.filter(pl.col("horizon_a") != pl.col("horizon_b")),
            {
                "horizon_a": "Horizon A",
                "horizon_b": "Horizon B",
                "valid_pairs": "Valid pairs",
                "correlation": "Pearson correlation",
            },
        ),
        "These returns ignore barriers and require all scheduled minutes through their horizon. "
        "Correlations use pairwise-complete windows; overlapping targets share price paths "
        "and are not independent observations. Fixed-horizon return regression is an appropriate "
        "benchmark alongside direct exit-policy return regression and multiclass labels.",
        "## 10. Initial candidates and next steps",
        f"Carry forward **{reference}/{h}m** as the specified-policy baseline; "
        f"**{reference}/{alternate['horizon']}m** as a duration contrast (expiry "
        f"{f(alternate['expire_pct'])}%); and **{third}/{h}m** as a barrier-width contrast "
        f"(expiry {f(third_row['expire_pct'])}%, ambiguity {f(third_row['ambiguous_pct'])}%). "
        "These candidates span observed label behavior; they are not ranked by gross mean "
        "or selected for balanced classes. Wider barriers' expiry rates and tighter barriers' "
        "resolution/ambiguity rates quantify their effective tightness; neither alone "
        "disqualifies a target.",
        "Before a training dataset, audit missing/invalid windows and raw-price/corporate-action "
        "flags, adopt the prior-only universe population fix, and decide treatment of historical "
        "asset availability. Retain ambiguity and UNKNOWN masks. Compare the classifier plus "
        "conditional expiration regressor with direct policy-return and fixed-horizon regression "
        "under chronological validation, purging overlapping windows. Keep 2024 onward reserved. "
        "Any execution-policy refinement must be checked separately; these are research fill "
        "assumptions.",
        "## Runtime and reproducibility",
        f"Elapsed through aggregation: {q['analysis_elapsed_seconds']:.2f}s; extraction batches "
        f"{q['extraction_batch_seconds']:.2f}s; measured peak process working set at that point "
        f"{q['peak_process_mb']:.1f} MB. Final runtime including charts, final peak, and output "
        f"bytes "
        "are in data_quality.json. Source/dependency hashes and the manifest signature are "
        "in provenance.json. Session-quality JSON files include individual batch timings and "
        "counts.",
        "Monthly request files contain multiple symbols and many sessions; the layout is not "
        "session-partitioned. Manifest symbol/date selection narrows files; DuckDB filters "
        "columns, regular hours, symbols, and timestamps with Parquet pruning. One symbol/session "
        "path is scanned for distinct thresholds once, shared across policies and durations. "
        "Only compact first-touch/fixed-return checkpoints persist. Exact quantiles aggregate "
        "one policy at a time within the configured DuckDB budget; no raw data is rewritten.",
        'Reproduce with `pip install -e ".[dev,research]"` then '
        "`dead-reckoning --config configs/label_study_v1.yaml research labels`. "
        "For a bounded trial add `--smoke`; to resume use the same configuration/mode and "
        "`--resume <run-directory>`. Resume rejects source, dependency, configuration, or "
        "manifest changes.",
    ]
    (output / "report.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
