# Trophy Rarity Forecast v10

[![make.fun entry](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v10/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v10/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty on make.fun: forecast every GTA VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

## How it works

A trophy's earn rate is earners over tracked players, and the two move in
opposite directions: players keep earning trophies (the rate climbs) while
Exophase keeps refreshing its tracked-player count (new players dilute the
rate). Launch night was frozen at 638 tracked players and then refreshed
to 3,868, so a forecast that just repeats the last rate is wrong on purpose.

The program (`main.py`, Python 3.13, standard library only) picks one of
three regimes from the shape of the case history:

* **Refresh** — frozen history, short horizon, last read still at 638
  players: rebuild the refreshed earner counts directly. The prologue
  trophy absorbs almost every new player; reachable trophies end at twice
  their old earners minus two; the rest are untouched.
* **Prior** — frozen history, longer horizon: description-based priors on
  the logit scale. Early story beats rise (+1.3 logit/day), mid beats rise
  a little (+0.4), everything else falls hard (-3.0), with week-one
  ceilings per trophy class (early 40%, activities 25%, mid 5%,
  late/grind 1%).
* **Fitted** — moving history: fit each trophy's earners and player count
  with a power curve and a saturating curve, combine them geometrically,
  and divide. A straight logit trend over the last 48 reads is the
  fallback when a trophy has too little history to fit.

Quantiles come from the median on the logit scale. In the refresh branch
the spread is 0.5 logit units — wider than the 0.3 used by the earlier
v5 entry, because the two live practice grades showed the true rates
deviating from the median further than the model assumed. Everywhere else
the spread grows with the square root of the horizon plus the fit's
residual, capped at 2.5.

## A detail worth recording

The refresh rule was reconstructed from the earlier entry's own output
numbers: reachable trophies land at exactly twice their old earners minus
two, and the total then lands exactly on 4,276 with no trimming step at
all — the old earners across all trophies sum to 943, the page's own
`trophies_earned` count at the first read. A round-robin trim to 4,276 is
kept in the code only as a safety net that never fires on the practice
data.

## Running it

```sh
MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out`
per case. Deterministic, offline, about 0.07 s per case.

## Local verification

* Practice cases p00/p01: all 55 medians identical to the v5 entry's
  output (0 differences); quantiles wider on all 55 trophies in both
  cases (spread 0.5 vs 0.3).
* Output format: 55 trophies × 5 quantiles, ids 1–55 each once, all
  values finite and non-decreasing, 2,622 bytes per output.
* Determinism: two runs produce byte-identical output.
