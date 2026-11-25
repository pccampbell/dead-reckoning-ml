"""Exact quantiles from one policy at a time; no copies of the raw price dataset."""

import logging

import numpy as np
import polars as pl

from dead_reckoning.labels.barriers import Outcome, resolve

log = logging.getLogger(__name__)


def distribution_sql(column, prefix, *, scale=1, where="true"):
    value = f"CASE WHEN {where} THEN {column}*{scale} END"
    pieces = [
        f"avg({value}) AS {prefix}_mean",
        f"median({value}) AS {prefix}_median",
        f"stddev_samp({value}) AS {prefix}_std",
        f"100.0*count(*) FILTER (WHERE {where} AND {column}>0) / "
        f"nullif(count({value}),0) AS {prefix}_positive_pct",
        f"100.0*count(*) FILTER (WHERE {where} AND {column}<0) / "
        f"nullif(count({value}),0) AS {prefix}_negative_pct",
    ]
    for q in [5, 25, 75, 95]:
        pieces.append(f"quantile_cont({value},{q / 100}) AS {prefix}_p{q}")
    return pieces


def policy_metrics_sql():
    fields = [
        "count(*) AS candidates",
        "count(ret) AS valid",
        "count(*) FILTER (WHERE outcome != 4 AND outcome != 3) AS excluded_valid",
    ]
    for outcome in Outcome:
        name = outcome.name.lower()
        fields.extend(
            [
                f"count(*) FILTER (WHERE outcome={int(outcome)}) AS {name}_count",
                f"100.0*count(*) FILTER (WHERE outcome={int(outcome)})/count(*) AS {name}_pct",
            ]
        )
    fields.extend(
        [
            "100.0*count(*) FILTER (WHERE outcome IN (0,1,3))/nullif(count(ret),0) "
            "AS resolved_valid_pct",
            "count(*) FILTER (WHERE full_window) AS full_window_candidates",
            "100.0*count(*) FILTER (WHERE full_window AND outcome IN (0,1,3)) / "
            "nullif(count(*) FILTER (WHERE full_window),0) AS full_resolved_pct",
            "count(*) FILTER (WHERE outcome IN (1,3)) AS conservative_sl_count",
            "count(*) FILTER (WHERE outcome=0) AS conservative_tp_count",
            "count(*) FILTER (WHERE outcome IN (0,3)) AS optimistic_tp_count",
            "count(*) FILTER (WHERE outcome=1) AS optimistic_sl_count",
            "100.0*count(*) FILTER (WHERE outcome IN (1,3))/nullif(count(ret),0) "
            "AS conservative_sl_valid_pct",
            "100.0*count(*) FILTER (WHERE outcome=0)/nullif(count(ret),0) "
            "AS conservative_tp_valid_pct",
            "100.0*count(*) FILTER (WHERE outcome IN (0,3))/nullif(count(ret),0) "
            "AS optimistic_tp_valid_pct",
            "100.0*count(*) FILTER (WHERE outcome=1)/nullif(count(ret),0) "
            "AS optimistic_sl_valid_pct",
            "avg(optimistic)*10000 AS optimistic_mean_bps",
            "avg(CASE WHEN outcome!=3 THEN ret END)*10000 AS excluded_mean_bps",
            "sum(ret)*10000/nullif(sum(holding),0) AS mean_bps_per_holding_minute",
            "avg(ret)*10000 AS break_even_cost_bps",
            "sum(CASE WHEN outcome=2 THEN ret ELSE 0 END)*10000/nullif(count(ret),0) "
            "AS expiration_contribution_valid_bps",
            "sum(CASE WHEN outcome=2 THEN ret ELSE 0 END)*10000/count(*) "
            "AS expiration_contribution_candidate_bps",
            "sum(ret)*10000/count(*) AS known_return_sum_per_candidate_bps",
        ]
    )
    fields.extend(distribution_sql("ret", "gross_bps", scale=10000))
    fields.extend(distribution_sql("ret", "expiration_bps", scale=10000, where="outcome=2"))
    fields.extend(distribution_sql("holding", "holding_minutes"))
    for outcome in list(Outcome)[:4]:
        code, name = int(outcome), outcome.name.lower()
        fields.extend(
            [
                f"avg(CASE WHEN outcome={code} THEN holding END) AS holding_{name}_mean",
                f"median(CASE WHEN outcome={code} THEN holding END) AS holding_{name}_median",
                f"quantile_cont(CASE WHEN outcome={code} THEN holding END,.95) "
                f"AS holding_{name}_p95",
            ]
        )
    return ",\n".join(fields)


def aggregate(db, output, options):
    from dead_reckoning.research.study import unpack

    files = sorted((output / "touches").glob("*.parquet"))
    db.read_parquet([str(p) for p in files], hive_partitioning=False).create_view("evidence")
    fixed = []
    correlations = []
    for h in options.horizons:
        stats = distribution_sql(f"f{h}", "return_bps", scale=10000, where=f"isfinite(f{h})")
        row = db.sql(
            f"SELECT count(*) AS candidates, count(*) FILTER (WHERE isfinite(f{h})) "
            f"AS valid, {','.join(stats)} FROM evidence"
        ).pl()
        fixed.append(row.with_columns(pl.lit(h).alias("horizon")))
        for other in options.horizons:
            if other >= h:
                correlations.append(
                    db.sql(f"""
                    SELECT {h} AS horizon_a, {other} AS horizon_b,
                        count(*) FILTER (WHERE isfinite(f{h}) AND isfinite(f{other}))
                            AS valid_pairs,
                        corr(CASE WHEN isfinite(f{h}) AND isfinite(f{other}) THEN f{h} END,
                             CASE WHEN isfinite(f{h}) AND isfinite(f{other}) THEN f{other} END)
                             AS correlation FROM evidence
                """).pl()
                )
    pl.concat(fixed).write_csv(output / "fixed_horizon_summary.csv")
    pl.concat(correlations).write_csv(output / "fixed_horizon_correlations.csv")
    summary, segments, costs = [], [], []
    histograms = []
    expressions = policy_metrics_sql()
    dimensions = ["year", "time_of_day", "liquidity", "volatility"]
    for pair in options.pairs:
        for horizon in options.horizons:
            log.info("Research aggregation: %s / %sm", pair.name, horizon)
            db.execute("DROP TABLE IF EXISTS policy")
            for index, path in enumerate(files):
                # Read one compact session, then drop it; only one configuration in DuckDB.
                frame = pl.read_parquet(path)
                touches = unpack(frame, options)
                labels = resolve(touches, pair.tp, pair.sl, horizon)
                policy = (
                    frame.select(dimensions)
                    .with_columns(
                        pl.Series("outcome", labels.outcome),
                        pl.Series("ret", labels.conservative),
                        pl.Series("optimistic", labels.optimistic),
                        pl.Series("holding", labels.holding),
                        pl.Series("full_window", touches.first_bad >= max(options.horizons)),
                    )
                    .with_columns(pl.col(pl.Float64).fill_nan(None))
                )
                db.register("batch", policy.to_arrow())
                if index == 0:
                    db.execute("CREATE TEMP TABLE policy AS SELECT * FROM batch")
                else:
                    db.execute("INSERT INTO policy SELECT * FROM batch")
                db.unregister("batch")
            query = f"""
                SELECT CASE WHEN grouping(year)=0 THEN 'year'
                    WHEN grouping(time_of_day)=0 THEN 'time_of_day'
                    WHEN grouping(liquidity)=0 THEN 'liquidity'
                    WHEN grouping(volatility)=0 THEN 'volatility' ELSE 'all' END AS dimension,
                    coalesce(year,time_of_day,liquidity,volatility,'all') AS bucket,
                    {expressions}
                FROM policy GROUP BY GROUPING SETS (
                    (), (year), (time_of_day), (liquidity), (volatility)
                )
            """
            results = (
                db.sql(query)
                .pl()
                .with_columns(
                    pl.lit(pair.name).alias("pair"),
                    pl.lit(pair.tp).alias("tp"),
                    pl.lit(pair.sl).alias("sl"),
                    pl.lit(horizon).alias("horizon"),
                )
            )
            main = results.filter(pl.col("dimension") == "all")
            counts = [main[f"{o.name.lower()}_count"].item() for o in Outcome]
            if sum(counts) != main["candidates"].item():
                raise RuntimeError("Outcome counts do not sum to candidates")
            known = main["valid"].item()
            if known != main["candidates"].item() - counts[-1]:
                raise RuntimeError("Valid denominator differs from all non-UNKNOWN outcomes")
            summary.append(main)
            segments.append(results.filter(pl.col("dimension") != "all"))
            value = main.row(0, named=True)
            for cost in options.costs_bps:
                costs.append(
                    {
                        "pair": pair.name,
                        "horizon": horizon,
                        "cost_bps": cost,
                        "valid": known,
                        "conservative_net_mean_bps": value["gross_bps_mean"] - cost
                        if known
                        else None,
                        "optimistic_net_mean_bps": value["optimistic_mean_bps"] - cost
                        if known
                        else None,
                        "excluded_net_mean_bps": value["excluded_mean_bps"] - cost
                        if value["excluded_mean_bps"] is not None
                        else None,
                    }
                )
            hist = (
                db.sql("""
                SELECT floor(ret*10000/5)*5 AS bin_left_bps, count(*) AS count
                FROM policy WHERE outcome=2 GROUP BY 1 ORDER BY 1
            """)
                .pl()
                .with_columns(pl.lit(pair.name).alias("pair"), pl.lit(horizon).alias("horizon"))
            )
            histograms.append(hist)
    result = pl.concat(summary).sort("pair", "horizon")
    for group in result.partition_by("pair"):
        counts = group["full_window_candidates"].to_numpy()
        resolved = group["full_resolved_pct"].to_numpy()
        if not np.all(counts == counts[0]) or np.any(np.diff(resolved) < -1e-9):
            raise RuntimeError("Fully observed common-cohort first-resolution invariant failed")
    result.write_csv(output / "summary.csv")
    result.select("pair", "horizon", "full_window_candidates", "full_resolved_pct").write_csv(
        output / "resolution_curve.csv"
    )
    pl.concat(segments).sort("pair", "horizon", "dimension", "bucket").write_csv(
        output / "segment_summary.csv"
    )
    pl.DataFrame(costs).write_csv(output / "cost_sensitivity.csv")
    pl.concat(histograms).write_csv(output / "expiration_histogram.csv")
    db.execute("DROP TABLE policy")
