# Fantasy Scoring Notes

GridironIQ currently reconstructs the user's custom Yahoo scoring from nflverse
weekly aggregate data.

## Included

- +0.5 per completion
- 0.04 per passing yard
- cumulative passing-yard bonuses at 300/400/500
- +4 passing TD
- -2 interception
- +1 per 40+ yard completion
- 0.1 per rushing yard
- cumulative rushing bonuses at 100/150/200
- +6 rushing TD
- +2 per 40+ rush
- full PPR
- 0.1 per receiving yard
- cumulative receiving bonuses at 100/150/200
- +6 receiving TD
- +2 per 40+ reception
- +2 two-point conversion
- -2 fumble lost
- return yards / return TDs when the source fields are present

## Not Yet Reconstructible From Weekly Aggregate Data

The league separately awards +1 for a 40+ yard passing TD, rushing TD, or
receiving TD.

The weekly nflverse aggregate file tells us the number of 40+ yard plays and the
number of touchdowns, but it does not identify whether a specific long play was
also a touchdown. Those separate bonuses therefore require play-by-play data and
are intentionally omitted from the Phase 1 target instead of being guessed.
