# Candidate and label datasets

Stages are independent: **candidates → labels / feature groups → assembly → training**.
Only candidates and labels are implemented. Future features join on
`(symbol, decision_timestamp_utc)` within a referenced candidate artifact. No feature
store, scheduler, distributed runtime, or additional downloads are involved.

## Storage and identity

All outputs live under configurable `data.root`:

```text
datasets/
  candidates/<candidate_id>/
    metadata.json
    writer.duckdb
    session_date=YYYY-MM-DD/part.parquet
  labels/<barrier_label_id>/...
  labels/<fixed_horizon_id>/...
  features/<future_feature_group_id>/...   # convention only; not implemented
  assembled/<future_dataset_id>/...       # convention only; not implemented
```

One daily file contains all selected securities: normally 39,000 rows for 500
stocks at five-minute intervals, or 21,000 on a 210-minute early-close session.
This avoids per-symbol files and permits bounded session reads and selected-date
rebuilds. Parquet uses Zstandard compression and UTC microsecond timestamps.
When reading a glob with Polars, use `hive_partitioning=False` if necessary to
prevent inference of the already stored `session_date` column.

An artifact's fingerprint includes its complete semantic configuration, relevant
implementation hashes, schema version, and relevant dependency versions. Date
ranges and query resource settings are excluded from identity, allowing additional
dates and different resource budgets. Automatic IDs contain a fingerprint suffix;
explicit IDs must match the original definition or generation fails. Changing
barrier parameters does not change candidates or fixed-horizon definitions.
Changing fixed horizons does not change barrier definitions.

`metadata.json` records configuration, source candidate ID/definition, schema,
creation time, per-session completion states, source dependencies, row counts,
validity summaries, Parquet checksums/sizes and build-run measurements. Runs record
requested ranges and resources. Coverage envelopes may contain holes; the
per-session states are authoritative. `writer.duckdb` supplies an OS-level writer
lock, including on Windows; it is not an artifact registry or data service.

Writes mark a partition building before an atomic Parquet replacement, then mark
it complete. A crash before completion causes that session to be regenerated.
Completed files are checksum-verified before reuse. Source candidate checksums,
relevant minute-request records and raw file size/mtime fingerprints detect stale
label inputs. Raw input checks are lightweight fingerprints, not a full content
hash of billions of rows. Changed dependencies require explicit rebuilding or a
new artifact. Unrelated completed sessions remain usable. Source files must not
be modified concurrently with a build.

Candidate dependencies contain the actual prior-universe rows and session bounds,
not minute prices or future availability. A corrected-universe generation change
with identical selected rows does not invalidate existing candidate partitions.
The builder checks that its universe generation matches the daily ingestion
fingerprint before use; it never silently substitutes the older same-day universe.

## Candidates

Defaults come from [dataset_v1.yaml](../configs/dataset_v1.yaml): top 500 by prior
20-observation median dollar volume, at least 20 observations, last prior daily
bar within seven calendar days, regular XNYS sessions, five-minute decisions,
one-minute assumed execution delay, long only. Calendar holidays, daylight saving
and early closes come from the installed exchange calendar.

The corrected `prior_session_asof_v2` curated universe is required and already
available in the development data root. It contains full daily ranks, so selecting
top 500 does not require rebuilding its historical top-2,000 union. If daily data
or universe settings change, first refresh the universe with the existing offline
`universe build` command. No provider client is created by dataset commands.

Every scheduled decision before the close is included, without a longest-horizon
cutoff. Missing entry prices, later missing bars and trade outcomes cannot change
eligibility. Missing completed minute requests cause the build to fail, naming
symbols; eligible symbols are never silently removed. A complete request with no
returned bars still generates candidates and later produces explicit invalid labels.

| Candidate field | Type / meaning |
|---|---|
| `symbol`, `decision_timestamp_utc` | String, UTC datetime; stable unique key |
| `session_date` | Date in the New York trading calendar |
| `decision_minute` | UInt16 offset from the session open |
| `session_open_utc`, `session_close_utc` | UTC datetime bounds |
| `universe_rank` | UInt32 prior liquidity rank |
| `prior_liquidity` | Float64 median dollar volume |
| `universe_feature_date` | Date of last strictly prior daily observation |

Execution delay is a candidate convention recorded in metadata. Label policies
must use the same delay. To change it, create a new candidate definition. Decisions
are at the scheduled grid instant; future feature code must use information
available at or before that instant under its own documented availability rules.

## Barrier targets

The default `original_60m` uses +0.50% TP, -0.40% SL, 60 elapsed minutes,
two consecutive missing interior minutes, and conservative ambiguity treatment.
`original_30m` and `wide_60m` (+1.00%/-0.80%, 60 minutes) are configured as well.
Select any subset together to share historical extraction and distinct first-touch
barrier scans. Thresholds remain configuration, not optimized research choices.

Entry is at the genuine valid entry-minute open. Offset 0 is that bar, and its
high/low participates in the path. An H-minute expiration uses the close of the
Hth bar, offset H-1, with exit time `entry_timestamp + H minutes`. For example,
a 10:00 decision, one-minute delay and 60-minute horizon enter at 10:01 and
expire at the close of the 11:00 bar, timestamped 11:01. The entire scheduled
window must fit the session, even if a hypothetical early barrier could have
been reached; eligibility cannot depend on that future event.

Opening prices precede high/low tests. An adverse stop gap fills at the observed
open and can lose more than the nominal SL. Favorable TP gaps fill at the TP
threshold. Intrabar exits use minute end as a holding-time/timestamp approximation;
OHLC cannot establish the exact touch instant. Gross returns exclude subsequent
round-trip transaction costs.

Absent bars are never synthesized or forward-filled. Multiple isolated gaps may
pass, but a consecutive interior run exceeding tolerance before an observed exit
invalidates the label. Genuine entry bars and, when no barrier exit is observed,
expiration bars are mandatory. Invalid OHLC, duplicates and unaligned bars are
always disqualifying before exit. Later gaps do not affect established exits.
Tolerance 0 retains strict behavior; YAML `null` allows unlimited interior gaps
for sensitivity experiments. Invalid reasons use the first chronological
disqualifier, with insufficient scheduled time taking precedence.

| Barrier field | Type / meaning |
|---|---|
| Keys and `session_date` | Same keys as source candidates, one row per candidate |
| `entry_timestamp_utc`, `entry_price` | Intended scheduled entry; genuine valid open as Float64, else null |
| `exit_timestamp_utc`, `exit_price` | Interpreted execution time and Float64 fill; null for invalid targets |
| `raw_outcome` | TP_FIRST, SL_FIRST, EXPIRE, AMBIGUOUS or UNKNOWN |
| `outcome` | Configured interpretation: TP_FIRST, SL_FIRST or EXPIRE; null if excluded/invalid |
| `conservative_outcome` | Always SL_FIRST for a valid ambiguous path |
| `realized_trade_return` | Float32 gross target under chosen interpretation; invalid is null, never zero-filled |
| `conservative_trade_return`, `optimistic_trade_return` | Float32 alternatives preserving ambiguity sensitivity |
| `expiration_return` | Float32, populated only for valid EXPIRE outcomes |
| `holding_minutes` | Float32 elapsed simulated holding duration |
| `label_valid`, `path_valid` | Boolean target usability and raw path validity |
| `invalid_reason` | valid, missing_entry, excessive_interior_gap, missing_expiration, invalid_data, insufficient_session_time, ambiguous_excluded |
| `ambiguous_outcome`, `stop_gap` | Boolean unresolved intrabar ordering and adverse stop-opening gap |
| `missing_minutes`, `max_consecutive_missing_minutes` | UInt16 path diagnostics through the first observed exit or attempted expiration |
| `permitted_interior_gap`, `strict_valid` | Boolean gap-bearing valid path and usability under tolerance 0 |

Barrier arithmetic and prices remain Float64 until generated return targets are
stored as Float32. Ambiguity never uses candle direction. With `ambiguity: exclude`,
raw evidence and both alternative returns remain available, while the configured
outcome, execution fields and targets are null and `label_valid` is false.
`path_valid` still distinguishes exclusion from missing data. Diagnostics for
invalid paths are descriptive; simulated outcomes/returns are never established.
For insufficient scheduled time, missing-minute diagnostics are null: unavailable
scheduled time is not a vendor data-completeness finding.

These fields support multiclass outcome prediction, expiration-only return
regression and direct trade-return regression. Future-path diagnostics must never
become entry-time features. Dropping invalid labels changes the sample population;
track liquidity/year/symbol coverage. A permitted gap can conceal a barrier touch
or reverse TP/SL ordering, so strict-versus-gap-bearing validation remains useful.

## Independent fixed-horizon targets

A separate artifact initially contains 15/30/60/90/120-minute returns over the same
keys and entry fields. Each H has `return_Hm` (Float32), `valid_Hm` (Boolean),
`invalid_reason_Hm` (String), and `exit_timestamp_Hm_utc` (UTC datetime). The exit
timestamp describes the scheduled endpoint if the window fits, including when
its price is missing. Entry and exit bars must be genuine, valid, unique and aligned;
intermediate coverage does not affect this target. Consequently a usable 15-minute
return can coexist with a scheduled-ineligible 120-minute target, or usable fixed
returns with UNKNOWN barrier labels. It is a separate target, not proof of
barrier execution quality.

## Commands, resumption and additional ranges

```powershell
# Stage 1: independent candidates over the bounded configured January 2019 range.
dead-reckoning --config configs/dataset_v1.yaml dataset candidates --output-id foundation_2019_v2

# Stage 2: primary policy and all configured fixed horizons.
dead-reckoning --config configs/dataset_v1.yaml dataset labels --candidate-id foundation_2019_v2 --policy original_60m --output-id foundation_original_60m_v2 --fixed-id foundation_fixed_v2

# Another policy, without regenerating candidates or fixed targets.
dead-reckoning --config configs/dataset_v1.yaml dataset labels --candidate-id foundation_2019_v2 --policy original_30m --no-fixed

# Multiple policies in one extraction; each gets its own artifact ID.
dead-reckoning --config configs/dataset_v1.yaml dataset labels --candidate-id foundation_2019_v2 --policy original_30m --policy wide_60m --no-fixed

# Fixed returns alone; completed partitions are skipped.
dead-reckoning --config configs/dataset_v1.yaml dataset labels --candidate-id foundation_2019_v2 --fixed-only --fixed-id foundation_fixed_v2

dead-reckoning --config configs/dataset_v1.yaml dataset status
```

Repeat the identical command to resume; completed partitions are verified and
skipped. To rebuild one date, use `--start 2019-01-31 --end 2019-02-01 --rebuild`
on the applicable stage with the same artifact ID. Only that session is replaced.
If candidate rows changed, rebuild dependent target partitions explicitly.
Append another range with `--start`/`--end` and the same definition/ID; earlier
partitions remain untouched. Feature groups will use the same dependency convention.
Status reports registered partitions and missing output files; generation performs
the stronger dependency/content checks.

To introduce a new TP, SL, tolerance, duration or ambiguity interpretation, add a
named policy in YAML and select `--policy <name>`. Omit `--output-id` for an automatic
new version or supply a fresh ID; incompatible existing IDs are rejected.

When the initial training period is chosen, the full-range commands are:

```powershell
dead-reckoning --config configs/dataset_v1.yaml dataset candidates --output-id <candidate-id> --start <start-date> --end <exclusive-end-date>
dead-reckoning --config configs/dataset_v1.yaml dataset labels --candidate-id <candidate-id> --start <start-date> --end <exclusive-end-date> --policy original_60m
```

No full-range build is launched by this implementation. Validation remains before
2024. Chronological splits, overlap purging, features, assembly, training and
backtesting are subsequent work. Existing raw corporate-action conventions and
historical asset-master limitations still apply; these labels do not correct them.

## Research compatibility

`labels/barriers.py` remains the first-touch/execution implementation.
`labels/paths.py` adds shared scheduled-fit, gap-quality and endpoint handling.
The missing-bar follow-up delegates to it rather than keeping a competing
calculation. `storage/history.py` supplies the existing bounded DuckDB extraction
to both research and datasets; original research import names remain available.

The old study's common cohort kept room for 120 minutes at every entry. The new
candidate grid does not, so full new-grid aggregate distributions are expected to
differ. Compare the reusable engine only on the saved research candidates when
checking numerical compatibility. The follow-up can write to a separate directory:

```powershell
python -m dead_reckoning.research.missing_bars <saved-study-directory> --output <new-verification-directory>
```

Original reports, saved candidates, first-touch checkpoints and research settings
are preserved. Resume correctly rejects checkpoints with changed source provenance;
re-running the saved settings in a new directory retains the historical strict
study convention. Dataset two-minute defaults do not alter those research defaults.
