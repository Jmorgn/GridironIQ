# Scheme-aware matchup modeling — future GridironIQ research

**Status:** roadmap / data-feasibility research. Not part of the present QB, RB,
or WR production rankings.

## Question

Can GridironIQ use each team's offensive tendencies, a player's success on
different route or play concepts, and an opponent's defensive weaknesses against
those concepts to improve pregame fantasy projections and defensive evaluation?

Hypothetical: if a receiver earns a high share of targets on slants, and the
opposing defense has historically allowed above-expected production on similar
routes, the matchup could increase the receiver's modeled opportunity or
efficiency. This is a **hypothetical example**, not a verified claim about any
specific player's route distribution.

## Possible modeling layers

1. **Offensive tendency:** neutral-situation pass rate, personnel groupings,
   formation, motion, route/play concept frequency, down/distance, red-zone
   tendencies, game script, offensive coordinator and QB tendencies.
2. **Player usage and suitability:** participation, routes run, targets per
   route, target share by route concept, route depth, yards per route run,
   separation and efficiency *when measured reliably*, receptions/expected
   points and touchdown opportunities.
3. **Defensive behavior:** coverage shell/personnel frequency, man/zone usage,
   blitz and pressure tendencies, defensive alignment, and allowed opportunity
   and efficiency against comparable route/play concepts.
4. **Context and interaction:** expected play volume, offensive line/pass
   protection, QB timing, probable coverage adjustment, injury-related
   personnel changes, down/distance and game script.
5. **Defensive-player extensions:** a defender's assignment, coverage/pressure
   opportunities, alignment, play-type susceptibility, expected tackles,
   pressures, sacks, turnovers, and missed-tackle risk where the data permits.

The model should estimate a *distribution* of likely plays and usage, not
assume the coordinator will repeatedly call the concept with the largest
historical matchup advantage.

## Data needed and limitations

- Traditional play-by-play and game-level nflverse data is a useful starting
  point for team play selection, pass/rush tendencies, down/distance, game
  script and outcome modeling.
- Reliable exact route concepts (e.g., slant vs dig), route participation,
  receiver assignment, primary read and defensive coverage assignments are
  generally **not consistently available in ordinary play-by-play tables**.
  Detailed route/coverage features require appropriately licensed or otherwise
  accessible tracking or charted data and careful identity/play joins.
- Do not mislabel a short completion as a slant or treat pass direction as an
  exact route concept. Proxies should remain explicitly labeled as proxies.
- Verify the actual coverage, historical seasons, licensing and as-of timing
  of any proposed dataset before building around it.
- Small concept-versus-defense samples, opponent quality, route combinations,
  coverage disguise, line pressure, game context and coordinator adjustments
  can all create misleading apparent matchup advantages.

## Staged implementation

**Stage A — public-data baseline:** add opponent-adjusted, *pregame* team
pass/rush tendencies by situation and comparable defensive tendencies. Compare
their incremental value against the current models, avoiding duplicated
signals from opponent aggregate rankings.

**Stage B — play/route charting:** if suitable data is available, build
receiver × concept × coverage and defense × concept histories. Use hierarchical
shrinkage / partial pooling for rare matchups. Preserve player/team continuity
through trades, QB changes, and coordinator changes.

**Stage C — scheme-matchup model:** estimate concept usage and player target
opportunity, then expected efficiency conditional on the defense. Feed
uncertainty-aware summary features or a separate matchup prediction into
GridironIQ.

**Stage D — defensive projections:** adapt scheme and assignment features
for team defense or individual defensive-player scoring if relevant to the
league and supported by the scoring configuration.

Every stage must use data available **before kickoff**, be evaluated with
walk-forward seasons, report both overall and lineup-relevant error, and keep
2026 live results out of retrospective model selection.

## Current priority

Finish `src/models/experiment_wr_two_conditional.py` and evaluate whether
modeling fantasy production both above and below the 65% snap threshold
improves unseen-season predictions. Do not promote the experiment automatically.
