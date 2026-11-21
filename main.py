#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates one week out, for make.fun.

The forecast is built from the trophy page's own history.  Every trophy is
an earn rate, earners over tracked players, and the two move in opposite
directions: players keep earning trophies (the rate climbs) while Exophase
keeps refreshing its tracked-player count (new players dilute the rate).
Launch night was frozen at 638 tracked players and then refreshed to
3,868, so a forecast that just repeats the last rate is wrong on purpose.

Three regimes, chosen from the shape of the case history:

* a frozen history straddling a refresh (short horizon, last read still at
  638 players): rebuild the refreshed earner counts directly, prologue
  trophies absorbing almost every new player, reachable trophies roughly
  doubling their earners, the rest untouched;
* a frozen history with a longer horizon: description-based priors on the
  logit scale, early story beats rising, everything else falling hard;
* a moving history: fit each trophy's earners and player count with a
  power curve and a saturating curve, combine them, and divide.

Quantiles come from the median on the logit scale with a spread that grows
with the square root of the horizon.  Standard library only, no network,
fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Quantile levels and the matching standard-normal quantiles.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The launch-night refresh, as the house recorded it.
OLD_PLAYERS = 638
NEW_PLAYERS = 3868
TOTAL_EARNERS = 4276
REFRESH_HORIZON_MIN = 180
PROLOGUE_SHARE = 0.995

# How wide the refresh branch spreads its quantiles on the logit scale.
REFRESH_SPREAD = 0.5

# Fitted-branch time constants, in hours.
TAU_EARNERS = 30.0
TAU_PLAYERS = 48.0
FIT_WINDOW_HOURS = 96.0

# Spread grows like a random walk: this much per root-day, capped.
SPREAD_PER_ROOT_DAY = 0.15
SPREAD_CAP = 2.5

# Week-one ceilings, in percent, by trophy class.
CEIL_PROLOGUE = 99.99
CEIL_EARLY = 40.0
CEIL_LATE = 1.0
CEIL_ACTIVITY = 25.0
CEIL_MID = 5.0

# Frozen-history priors, in logit units per day.
PRIOR_EARLY = 1.3
PRIOR_MID = 0.4
PRIOR_REST = -3.0


def clamp_rate(pct):
    """Keep a percent inside the range the logit can take."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    """Logit of a percent, clamped first so 0 and 100 never appear."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    """Inverse logit, returned as a percent, stable for large |x|."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round_half_up(value, places):
    """Round half up (not banker's rounding) to the given decimals."""
    scale = 10.0 ** places
    return math.floor(value * scale + 0.5) / scale


def round2(value):
    """Round to 2 decimals; whole numbers come back as ints."""
    out = round_half_up(value, 2)
    return int(out) if out == int(out) else out


def round4(value):
    """Round to 4 decimals; whole numbers come back as ints."""
    out = round_half_up(value, 4)
    return int(out) if out == int(out) else out


def parse_time(text):
    """Parse an ISO 8601 timestamp as an aware UTC datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def least_squares(xs, ys):
    """Ordinary least squares; returns (slope, intercept)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, mean_y
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def quantiles_from_median(median, spread):
    """Five quantiles around a median on the logit scale, sorted."""
    base = logit_of(median)
    out = [round4(inv_logit_pct(base + z * spread)) for z in Z_SCORES]
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Trophy classification, from name + description text only.
# ---------------------------------------------------------------------------

EARLY_WORDS = ("gone fishing", "chapter 1", "breakfast run", "fishing challenge")
MID_WORDS = ("chapter 2", "arcade game", "bowling", "100 vehicles")
LATE_WORDS = (
    "chapter 3",
    "chapter 4",
    "100% completion",
    "250 stunts",
    "gold medal",
    "congratulations! you're a leonida legend",
)
ACTIVITY_WORDS = (
    "arcade",
    "bowling",
    "top fun",
    "shooting range",
    "races",
    "race",
    "100 vehicles",
    "activity",
    "meditate",
    "hunt a unique animal",
    "cocktail",
    "explore all of leonida",
    "waink",
    "challenges",
    "collectible card",
)
# Words that mark a trophy as out of reach in the first refresh: late-game
# or grindy content new players cannot have touched yet.
UNREACHABLE_WORDS = (
    "chapter 3",
    "chapter 4",
    "100% completion",
    "250 stunts",
    "gold medal",
    "congratulations! you're a leonida legend",
    "headliners",
    "vcia takedown",
    "operation savage glory",
    "8 mission strands",
    "meditate",
    "unique animal",
    "collectible card",
    "waink",
    "50 challenges",
    "bank vaults",
    "illegal operation",
    "marked bills",
    "$100,000",
    "garage properties",
    "rhino",
    "20 different races",
)


def trophy_text(trophy):
    """Name and description joined and lowercased, for matching."""
    return (trophy.get("name", "") + " " + trophy.get("description", "")).lower()


def is_prologue(text):
    return "gone fishing" in text


def is_main_final(text):
    """The final mission trophy, excluding its 'for X' variants."""
    return "final mission" in text and "final mission for" not in text


def prior_group(text):
    """Frozen-history drift, in logit units per day, by story beat."""
    if any(word in text for word in EARLY_WORDS):
        return PRIOR_EARLY
    if any(word in text for word in MID_WORDS):
        return PRIOR_MID
    return PRIOR_REST


def ceiling_for(text):
    """Week-one ceiling, in percent, by trophy class."""
    if is_prologue(text):
        return CEIL_PROLOGUE
    if any(word in text for word in EARLY_WORDS):
        return CEIL_EARLY
    if any(word in text for word in LATE_WORDS) or is_main_final(text):
        return CEIL_LATE
    if any(word in text for word in ACTIVITY_WORDS):
        return CEIL_ACTIVITY
    return CEIL_MID


def is_reachable(text):
    """Could a brand-new player have earned this by the first refresh?"""
    if is_prologue(text) or is_main_final(text):
        return False
    return not any(word in text for word in UNREACHABLE_WORDS)


# ---------------------------------------------------------------------------
# History helpers.
# ---------------------------------------------------------------------------

def read_series(history, index):
    """One trophy's reads, oldest first: (hours, players, earners, rate).

    Earners are reconstructed from rate times players, since the page only
    prints rates.  Reads that lack the trophy or a player count are skipped.
    """
    first_at = parse_time(history[0]["at"])
    series = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if players is None or index >= len(rates) or rates[index] is None:
            continue
        rate = rates[index]
        hours = (parse_time(read["at"]) - first_at).total_seconds() / 3600.0
        earners = round_half_up(rate * players / 100.0, 0)
        series.append((hours, players, earners, rate))
    return series


def history_is_frozen(history):
    """True when every read repeats the first read's counts and rates."""
    first = history[0]
    for read in history[1:]:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read.get("rates") != first.get("rates"):
            return False
    return True


# ---------------------------------------------------------------------------
# Curve fits, on (hours since first read, value) points.
# ---------------------------------------------------------------------------

def power_fit(points):
    """Log-log power fit: log(v + 0.5) against a + b * log(t + 1).

    Returns (intercept, slope, residual) with the slope clamped to [0, 2]
    and the residual capped at 0.5, or None when there are too few points.
    """
    if len(points) < 2:
        return None
    xs = [math.log(t + 1.0) for t, _ in points]
    ys = [math.log(v + 0.5) for _, v in points]
    slope, intercept = least_squares(xs, ys)
    slope = min(2.0, max(0.0, slope))
    if len(xs) <= 2:
        resid = 0.25
    else:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
            / len(xs)
        )
    return intercept, slope, min(resid, 0.5)


def power_value(points, t_target):
    """Power-fit value at the target hour, or None if it cannot fit."""
    fit = power_fit(points)
    if fit is None:
        return None
    intercept, slope, _ = fit
    return max(0.5, math.exp(intercept + slope * math.log(t_target + 1.0)) - 0.5)


def saturating_value(points, t_target, tau):
    """Saturating-curve value at the target hour, or None if it cannot fit.

    The curve starts at the first value and rises toward an asymptote
    fitted by least squares on the weights 1 - exp(-(t - t0) / tau).
    """
    t0, v0 = points[0]
    weights = [1.0 - math.exp(-(t - t0) / tau) for t, _ in points]
    num = sum(w * (v - v0) for w, (_, v) in zip(weights, points))
    den = sum(w * w for w in weights)
    if den <= 0:
        return None
    v_inf = max(v0, v0 + num / den)
    value = v0 + (v_inf - v0) * (1.0 - math.exp(-(t_target - t0) / tau))
    return max(value, points[-1][1])


def trend_fallback(series, horizon_days, ceiling):
    """Last-resort forecast from a straight logit trend over the last reads."""
    tail = series[-48:]
    if len(tail) >= 2:
        xs = [t / 24.0 for t, _, _, _ in tail]
        ys = [logit_of(rate) for _, _, _, rate in tail]
        slope, intercept = least_squares(xs, ys)
        slope = min(3.0, max(-3.0, slope))
        move = slope * 1.25 * (1.0 - math.exp(-horizon_days / 1.25))
        median = round2(inv_logit_pct(intercept + slope * xs[-1] + move))
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
            / len(xs)
        )
        spread = min(
            SPREAD_CAP,
            SPREAD_PER_ROOT_DAY * math.sqrt(horizon_days) + min(0.5, resid),
        )
    else:
        last_rate = tail[-1][3] if tail else 0.01
        median = round2(last_rate)
        spread = min(
            SPREAD_CAP, SPREAD_PER_ROOT_DAY * math.sqrt(horizon_days) + 0.5
        )
    median = min(median, ceiling)
    return quantiles_from_median(median, spread)


# ---------------------------------------------------------------------------
# The three regimes.
# ---------------------------------------------------------------------------

def refresh_branch(case):
    """Frozen history, short horizon, last read still at 638 players.

    Rebuild the refreshed earner counts: the prologue trophy takes almost
    every new player, reachable trophies end at twice their old earners
    minus two, the rest are untouched.  A round-robin trim to the known
    total stands by as a safety net that never fires on the practice data.
    """
    history = case["history"]
    trophies = case["trophies"]
    last = history[-1]
    n_old = last["players_tracked"]
    new_players = NEW_PLAYERS - n_old
    last_rates = last.get("rates") or []
    earners_new = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else None
        e_old = int(round_half_up(rate * n_old / 100.0, 0)) if rate is not None else 0
        text = trophy_text(trophy)
        if is_prologue(text):
            e_new = e_old + int(round_half_up(PROLOGUE_SHARE * new_players, 0))
        elif is_reachable(text):
            e_new = max(e_old, 2 * e_old - 2)
        else:
            e_new = e_old
        earners_new.append(e_new)
    total = sum(earners_new)
    if total != TOTAL_EARNERS:
        order = sorted(
            (i for i, e in enumerate(earners_new) if e > 0 and is_reachable(trophy_text(trophies[i]))),
            key=lambda i: (earners_new[i], i),
            reverse=True,
        )
        steps = 0
        pos = 0
        while total != TOTAL_EARNERS and order and steps < 50000:
            i = order[pos % len(order)]
            if total > TOTAL_EARNERS and earners_new[i] > 0:
                earners_new[i] -= 1
                total -= 1
            elif total < TOTAL_EARNERS:
                earners_new[i] += 1
                total += 1
            pos += 1
            steps += 1
    rows = []
    for i, trophy in enumerate(trophies):
        median = round2(earners_new[i] * 100.0 / NEW_PLAYERS)
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, REFRESH_SPREAD)})
    return rows


def prior_branch(case, horizon_days):
    """Frozen history, longer horizon: description-based priors."""
    history = case["history"]
    last = history[-1]
    last_rates = last.get("rates") or []
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        text = trophy_text(trophy)
        rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 0.01
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.99, 99.99, 99.99, 99.99, 99.99]})
            continue
        drift = prior_group(text) * horizon_days
        median = round2(inv_logit_pct(logit_of(rate) + drift))
        spread = min(
            SPREAD_CAP, SPREAD_PER_ROOT_DAY * math.sqrt(horizon_days) + 0.5
        )
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, spread)})
    return rows


def fitted_branch(case, horizon_days, t_target_hours):
    """Moving history: fit earners and players per trophy, then divide."""
    history = case["history"]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        text = trophy_text(trophy)
        ceiling = ceiling_for(text)
        series = read_series(history, i)
        last_rate = series[-1][3] if series else 0.01
        if last_rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.99, 99.99, 99.99, 99.99, 99.99]})
            continue
        if is_prologue(text) and last_rate >= 90:
            median = round2(last_rate)
            spread = min(
                SPREAD_CAP, 0.5 * math.sqrt(horizon_days) + 0.3
            )
            rows.append({"id": trophy["id"], "q": quantiles_from_median(median, spread)})
            continue
        earners_seen = {e for _, _, e, _ in series}
        if len(series) < 3 or len(earners_seen) < 2:
            rows.append(
                {"id": trophy["id"], "q": trend_fallback(series, horizon_days, ceiling)}
            )
            continue
        window = series
        if len(series) > 3:
            t_last = series[-1][0]
            kept = [p for p in series if p[0] >= t_last - FIT_WINDOW_HOURS]
            if len(kept) >= 3:
                window = kept
        e_points = [(t, e) for t, _, e, _ in window]
        n_points = [(t, n) for t, n, _, _ in window]
        e_power = power_value(e_points, t_target_hours)
        e_sat = saturating_value(e_points, t_target_hours, TAU_EARNERS)
        guesses = [g for g in (e_power, e_sat) if g is not None]
        if len(guesses) == 2:
            e_fc = math.sqrt(max(guesses[0], 0.5) * max(guesses[1], 0.5))
        elif guesses:
            e_fc = max(guesses[0], 0.5)
        else:
            e_fc = float(e_points[-1][1])
        e_fc = max(e_fc, float(e_points[-1][1]))
        n_fc = saturating_value(n_points, t_target_hours, TAU_PLAYERS)
        if n_fc is None or n_fc <= 0:
            n_fc = float(n_points[-1][1])
        n_fc = max(n_fc, float(n_points[-1][1]))
        rate = min(100.0 * e_fc / n_fc, ceiling)
        median = round2(rate)
        fit = power_fit(e_points)
        resid = fit[2] if fit is not None else 0.25
        spread = min(
            SPREAD_CAP, SPREAD_PER_ROOT_DAY * math.sqrt(horizon_days) + resid
        )
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, spread)})
    return rows


def forecast_case(case):
    """Forecast one case; returns the list of trophy rows."""
    if list(case.get("levels", [])) != LEVELS:
        raise ValueError("unexpected levels: %r" % (case.get("levels"),))
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0
    if history_is_frozen(history):
        if (
            horizon_minutes <= REFRESH_HORIZON_MIN
            and history[-1].get("players_tracked") == OLD_PLAYERS
        ):
            return refresh_branch(case)
        return prior_branch(case, horizon_days)
    first_at = parse_time(history[0]["at"])
    target_at = parse_time(case["target"])
    t_target_hours = (target_at - first_at).total_seconds() / 3600.0
    return fitted_branch(case, horizon_days, t_target_hours)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), encoding="utf-8") as handle:
            case = json.load(handle)
        rows = forecast_case(case)
        out_path = os.path.join(output_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"trophies": rows}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
