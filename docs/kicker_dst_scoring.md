# Kicker and D/ST Scoring — GridironIQ

Source: the user's Yahoo league-setting screenshots supplied on
2026-10-02, and their explicit clarification that three unlisted
scoring categories are worth **zero**. No default scoring is inferred
for other categories. Python constants and validated helper functions:
[`src/scoring/league_rules.py`](../src/scoring/league_rules.py).

## Kicker (K)

| Event | Points |
| --- | ---: |
| Field goal made, 0–19 yards | 3 |
| Field goal made, 20–29 yards | 3 |
| Field goal made, 30–39 yards | 3 |
| Field goal made, 40–49 yards | 4 |
| Field goal made, 50+ yards | 5 |
| Field goal missed, 0–19 yards | -1 |
| Field goal missed, 20–29 yards | -1 |
| Field goal missed, 30–39 yards | -1 |
| Field goal missed, 40–49 yards | -0.5 |
| Field goal missed, 50+ yards (unlisted) | 0 |
| Point-after attempt made | 1 |
| Point-after attempt missed | 0 |
| Point-after attempt blocked | 0 |

The user explicitly confirmed that missed and blocked extra
points carry zero penalty. Missed field goals from 50+ yards
also have no scoring category and carry zero penalty.

Distance-bucket stats expected in local nflverse player-week files:
`fg_made_0_19`, `fg_made_20_29`, `fg_made_30_39`,
`fg_made_40_49`, `fg_made_50_59`, `fg_made_60_`,
the corresponding `fg_missed_*` columns, and `pat_made`.
The helper `kicker_known_components()` returns `fantasy_points`
using all confirmed league Kicker scoring categories. It also
retains the former `confirmed_component_points` key as a
backward-compatible alias. Field goals missed from 50+ yards
and missed/blocked extra points score zero.

## Defense / Special Teams (D/ST)

| Event | Points |
| --- | ---: |
| Sack | 0.5 |
| Interception | 2 |
| Fumble recovery | 2 |
| Defensive or special-teams touchdown | 6 |
| Safety | 2 |
| Blocked kick | 2 |
| Kickoff-return touchdown | 6 |
| Punt-return touchdown | 6 |
| Tackle for loss | 0.5 |
| Three-and-out forced | 1 |
| Extra point returned | 2 |

A touchdown event must be scored **once** in its proper category.
The separate return-TD lines describe eligible special-teams events,
not a second touchdown bonus.

### Points allowed

| Opponent points allowed | D/ST bonus |
| --- | ---: |
| 0 | 10 |
| 1–6 | 7 |
| 7–13 | 4 |
| 14–20 | 1 |
| 21–27 (unlisted) | 0 |
| 28–34 | -1 |
| 35+ | -4 |

### Defensive yards allowed

| Total yards allowed | D/ST bonus |
| --- | ---: |
| Negative | 4 |
| 0–99 | 3 |
| 100–199 | 2 |
| 200–299 | 1 |
| 300–399 (unlisted) | 0 |
| 400–499 | -1 |
| 500+ | -2 |

The shared scoring helper returns zero for these two explicitly
confirmed unlisted brackets, rather than substituting Yahoo defaults.

## Historical label quality: validate before modeling

Our existing downloader obtains nflverse player-week, team-week,
schedule, snap, depth-chart and injury files. It **does not**
currently obtain play-by-play. Aggregated NFL team statistics
may include some defensive statistics, but that does not establish
that they contain all categories in this custom league.

A D/ST training target must account for **three-and-outs**,
**tackles for loss**, **blocked kicks**, **return touchdowns** and
**returned extra points**, in addition to takeaways and basic
points/yards allowed. Some require play-level attribution or a
verified defensive team-stat source. In particular, Yahoo D/ST
points-allowed scoring must be verified against its definition
rather than blindly using an opponent's final schedule score
(which can include defensive/special-teams points by that opponent).
Similarly, net yards allowed must agree with the fantasy platform
and team-stat definitions.

The local audit also checks depth-chart K/PK rows to establish a
pregame candidate population before model training.

Run the local schema audit:

```cmd
py src\data\audit_k_dst_sources.py
py -m unittest discover -s tests -v
```

The audit checks selected 2021–2026 kicking fields, current
team defensive columns, examples of missing long-kick/PAT events,
and whether play-by-play has been downloaded. It creates **no
new training data or production models**.

**Modeling sequence:** build and validate the leakage-safe
Kicker v1 candidate dataset and 2023–2025 walk-forward benchmark;
select a Kicker architecture based on the results; separately
validate team-based D/ST labels with the missing play-level
statistics; integrate successful models into the weekly runner
and prospective tracker. The existing QB/RB/WR/TE and
RB/WR-only FLEX models are unaffected.
