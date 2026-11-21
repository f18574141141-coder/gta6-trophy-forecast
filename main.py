"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Method (standard library only, no network, deterministic).

An earn rate is a ratio: earned_trophies / players_tracked.  The two halves
move for different reasons -- new players join Exophase's tracking (the
denominator grows) while the existing players progress through the game
(the numerators grow) -- so this program forecasts them separately and
divides, instead of extrapolating the ratio itself:

* Players: least-squares line of log(players) against log(1 + hours since
  the release day began), fitted on the last 24 hours of distinct reads
  (launch-night dynamics no longer apply) and read at the target.
* Earned counts, per trophy: the same log-log line per trophy, each slope
  shrunk towards the mean slope over all trophies (a trophy's own
  history is short and noisy).  A trophy nobody has earned yet is
  forecast from half an earner, so its median stays near zero without
  breaking the log.
* Median rate = 100 * earned* / players*, but only a damped share of
  the logit move from the trophy's last read rate is kept: the further
  the target sits beyond the fitted window's span, the less of the
  fitted move is trusted.  Damping the move -- not the two halves
  separately -- keeps the ratio's own trend intact.

Exophase refreshes a game's page on its own schedule, in jumps: the
house saw the page frozen for hours on launch night and then jump to
six times the players tracked.  Two guards follow from that:

* When the history never changed at all (one distinct read), the page
  is certainly stale and the underlying player base certainly grew
  during launch week, so a dilution prior is applied: players are
  expected to have multiplied by up to 4x over the stale window plus
  the horizon, while earned counts grow only as players^0.25 (a new
  player has usually earned far fewer trophies than an early one),
  except for a trophy at ~100%, which every new player holds and so
  tracks players one for one.  Once the history shows real refreshes
  this prior switches off: the fitted trends then carry the movement.
* The spread is honest about that lumpiness: a random-walk term that
  grows with the horizon, the fit's residual (with leverage at the
  target), a refresh term (wide when no refresh was ever observed),
  and a small floor.  Quantiles are the median shifted by the normal
  quantiles times that spread, on the logit scale.

Input: every <id>.in in $MAKEFUN_INPUT_DIR (else /input).  Output: one
<id>.out in $MAKEFUN_OUTPUT_DIR (else /output), {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}.  No randomness anywhere, so two
runs on one case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as the house baseline does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x.

    Returns (intercept, slope, resid_sd, leverage_fn, mean_x, mean_y):
    predictions are best re-anchored at (mean_x, mean_y), so a changed
    slope still runs through the cloud of fitted points.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0, (lambda x_new: 1.0 / n), mx, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        var = sum(r * r for r in resid) / (n - 2)
        resid_sd = math.sqrt(var)
    else:
        resid_sd = 0.0

    def leverage(x_new):
        return 1.0 / n + (x_new - mx) ** 2 / sxx

    return intercept, slope, resid_sd, leverage, mx, my


def distinct_states(history):
    """Collapse consecutive identical reads, keeping each state's first read."""
    states = []
    last_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != last_key:
            states.append(read)
            last_key = key
    return states


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_trophies = len(trophies)
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))

    states = distinct_states(case["history"])
    n_states = len(states)
    # Fit on the recent window: the last 24 hours of distinct reads, or
    # all of them when fewer than 3 distinct reads fall in the window.
    # Old reads describe launch-night dynamics that no longer apply.
    cutoff = checkpoint.timestamp() - 24.0 * 3600.0
    window = [s for s in states if parse_time(s["at"]).timestamp() >= cutoff]
    if len(window) < 3:
        window = states
    span_h = max(
        1.0,
        (parse_time(window[-1]["at"]) - parse_time(window[0]["at"])).total_seconds() / 3600.0,
    )
    # Extrapolation damping: the further the target is beyond the span
    # the trend was fitted on, the less of the fitted move we trust.
    # It damps the forecast *rate* move (below), not each half of the
    # ratio: damping numerator and denominator separately would shrink
    # the rate's own trend twice over.
    damp = math.sqrt(span_h / (span_h + horizon_h))

    xs = [math.log1p(hours_since_release(parse_time(s["at"]))) for s in window]

    # --- players forecast -------------------------------------------------
    log_p = [math.log(s["players_tracked"]) for s in window]
    if len(window) >= 2 and len(set(log_p)) > 1:
        _pi, p_slope, _pr, _pl, p_mx, p_my = ols(xs, log_p)
        players_star = math.exp(p_my + p_slope * (x_target - p_mx))
    else:
        p_slope, p_mx, p_my = 0.0, xs[-1], log_p[-1]
        players_star = math.exp(log_p[-1])
    players_last = states[-1]["players_tracked"]

    # Jump volatility of players across observed refreshes, for the spread.
    if n_states >= 2:
        all_log_p = [math.log(s["players_tracked"]) for s in states]
        jumps = [
            abs(all_log_p[i] - all_log_p[i - 1])
            for i in range(1, n_states)
            if all_log_p[i] != all_log_p[i - 1]
        ]
        jump_vol = (sum(j * j for j in jumps) / len(jumps)) ** 0.5 if jumps else 0.0
    else:
        jump_vol = 0.0

    # --- per-trophy earned-count fits --------------------------------------
    # earned[i][j] = earned count of trophy i in window state j (or None).
    earned = []
    for i in range(n_trophies):
        series = []
        for s in window:
            rate = s["rates"][i]
            series.append(None if rate is None else rate * s["players_tracked"] / 100.0)
        earned.append(series)

    # Mean log-earned slope across trophies, for shrinkage.  What is
    # shrunk is the slope *difference* (earned slope minus players
    # slope), i.e. the rate's own trend: shrinking the earned slope
    # alone would drag every rate down whenever players grow, even for
    # a trophy whose rate is flat -- and near 100% a hair of spurious
    # drift costs a lot in logit terms.
    diffs = []
    fits = []
    for i in range(n_trophies):
        pts = [(x, math.log(e)) for x, e in zip(xs, earned[i]) if e is not None and e > 0]
        if len(pts) >= 2 and len({y for _, y in pts}) > 1:
            inter, slope, resid, lev, mx, my = ols([p[0] for p in pts], [p[1] for p in pts])
            fits.append((inter, slope, resid, lev, mx, my, len(pts)))
            diffs.append(slope - p_slope)
        else:
            fits.append(None)
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

    # --- flat-history refresh prior ---------------------------------------
    flat = n_states == 1
    stale_h = 0.0
    players_mult = 1.0
    if flat:
        stale_h = (checkpoint - parse_time(states[0]["at"])).total_seconds() / 3600.0
        # Launch-week player base multiplies while the page sits frozen.
        players_mult = min(4.0, math.exp(0.20 * (stale_h + horizon_h)))
        players_star = players_last * players_mult

    last = states[-1]
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        last_earned = last_rate * players_last / 100.0

        fit = fits[i]
        if fit is not None:
            _inter, slope, resid, lev, mx, my, n_pts = fit
            w = n_pts / (n_pts + 3.0)  # shrink the rate trend towards the mean
            slope_use = p_slope + (w * (slope - p_slope) + (1.0 - w) * mean_diff)
            earned_star = math.exp(my + slope_use * (x_target - mx))
            fit_sig = min(1.0, max(0.15, resid * math.sqrt(1.0 + lev(x_target))))
        else:
            earned_star = last_earned if last_earned > 0 else 0.5
            fit_sig = 0.5

        if flat:
            # New players hold far fewer trophies than early ones, except
            # for a trophy everyone holds, which tracks players exactly.
            gamma = 1.0 if last_rate >= 99.0 else 0.25
            earned_star = (last_earned if last_earned > 0 else 0.5) * (players_mult ** gamma)

        model_rate = clamp_pct(100.0 * earned_star / players_star)
        if flat:
            median = model_rate
        elif last_rate >= 99.99:
            # Censored at the boundary: every tracked player holds it,
            # and shrinkage must not invent a drift away from 100%.
            median = 99.99
        else:
            # Damp the forecast *move* from the last rate: the fitted
            # lines are read undamped at the target, then only a share
            # of the logit move from the last rate is kept.
            median = inv_logit(
                logit(last_rate) + damp * (logit(model_rate) - logit(last_rate))
            )
        median = clamp_pct(median)

        # --- spread --------------------------------------------------------
        rw = 0.5 * math.sqrt(horizon_d)
        refresh_sig = 0.9 if flat else min(0.6, 0.25 + jump_vol)
        sigma = math.sqrt(rw * rw + fit_sig * fit_sig + refresh_sig * refresh_sig + 0.15 * 0.15)
        sigma = min(2.0, max(0.35, sigma))

        med_logit = logit(median)
        qs = []
        prev = None
        for z in Z:
            q = clamp_pct(inv_logit(med_logit + z * sigma))
            q = round4(q)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
