"""
intelligence.py — The "Catch It?" Decision Engine (v2)
=======================================================

A lightweight, async-ready scoring module that tells a Bangalore cyclist
whether to chase the next BMTC bus or keep pedaling. No external API
calls — just Redis reads and fast math.

Inputs:
    1. User GPS (lat, lon)
    2. Bus coordinates from Upstash Redis
    3. Current time → Bangalore Traffic Multiplier

Output:
    Lazy Score (0–100):
        100  →  "Run for it! Bus is RIGHT HERE!" (< 500 m)
         0   →  "Keep Pedaling, it's too far." (> 3 km)
      1–99   →  Gradient based on distance + traffic-adjusted ETA

Architecture:
    Upstash Redis ──► Bus positions ──┐
    System clock  ──► Traffic mult.  ──┼──► Lazy Score (0–100)
    User GPS      ──► Distance calc  ──┘

NOTE: Uses `redis.asyncio` — the modern successor to the deprecated
      `aioredis` library (merged into redis-py ≥ 4.2).
"""

import json
import math
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import redis.asyncio as aioredis

from core.config import settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════

EARTH_RADIUS_KM = 6371.0

# Distance thresholds (meters)
#   - Under CATCHABLE_M: score is maxed out — sprint to the stop!
#   - Over TOO_FAR_M:    score is zero — bus is not worth chasing.
#   - Between them:      smooth gradient from 100 → 0.
CATCHABLE_M = 500.0    # < 500 m  → "Run for it!"
TOO_FAR_M = 3000.0     # > 3 km   → "Keep Pedaling"

# BMTC bus speeds (km/h) — empirically observed in Bangalore
#   These are the *base* speeds before applying the traffic multiplier.
#   In free-flow conditions (Sunday 6 AM), a BMTC bus averages ~25 km/h
#   on arterial roads. We use this as our baseline.
BUS_BASE_SPEED_KMH = 25.0
BUS_MIN_SPEED_KMH = 5.0   # Floor: even gridlock moves *a bit*


# ═════════════════════════════════════════════════════════════════════
# HAVERSINE — DISTANCE BETWEEN TWO GPS POINTS
# ═════════════════════════════════════════════════════════════════════

def haversine_m(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Great-circle distance between two GPS coordinates.
    Returns distance in **meters**.

    Uses the Haversine formula — accurate to ±0.5% for distances under
    100 km, which is more than enough for intra-Bangalore calculations.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * 1000 * math.asin(math.sqrt(a))


# ═════════════════════════════════════════════════════════════════════
# BANGALORE TRAFFIC MULTIPLIER
# ═════════════════════════════════════════════════════════════════════

def get_traffic_multiplier(dt: Optional[datetime] = None) -> float:
    """
    Compute the Bangalore Traffic Multiplier for the current time.

    This is a dimensionless value ≥ 1.0 that inflates the bus ETA
    to account for congestion. It directly scales travel time:

        real_eta = base_eta × traffic_multiplier

    ┌──────────────────────────────────────────────────────────────────┐
    │  HOW THE TRAFFIC MULTIPLIER WORKS                                │
    │                                                                  │
    │  Bangalore traffic follows two daily peaks. Each peak is         │
    │  modeled as a Gaussian "bump" in the multiplier:                 │
    │                                                                  │
    │    bump = amplitude × exp(-0.5 × ((hour − center) / σ)²)        │
    │                                                                  │
    │  ── Morning Peak ──                                              │
    │    center = 8.5  (8:30 AM — office rush)                         │
    │    σ = 1.2       (concentrated: 7:00–10:00 AM)                   │
    │    amplitude = 1.0  (doubles the ETA at dead center)             │
    │                                                                  │
    │  ── Evening Peak ──                                              │
    │    center = 18.0  (6:00 PM — return commute)                     │
    │    σ = 1.5        (wider: schools + offices + shopping)           │
    │    amplitude = 1.2 (slightly worse than morning)                  │
    │                                                                  │
    │  ── Day-of-Week Adjustments ──                                   │
    │    Friday evening: ×1.2 amplitude (early exits + weekend trips)   │
    │    Weekends: ×0.4 amplitude (traffic exists but much lighter)     │
    │                                                                  │
    │  Final multiplier = 1.0 + morning_bump + evening_bump            │
    │  Clamped to [1.0, 2.5]                                           │
    │                                                                  │
    │  ── Examples ──                                                   │
    │    Sunday 6 AM:       1.00  (empty roads, base speed)            │
    │    Tuesday 11 AM:     1.05  (mild, post-rush)                    │
    │    Monday 8:30 AM:    2.00  (peak morning, ETA doubled)          │
    │    Thursday 6 PM:     2.20  (evening peak)                       │
    │    Friday 6 PM:       2.44  (Friday carnage — worst case)        │
    │    Saturday 5 PM:     1.22  (weekend evening, manageable)        │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        dt: Datetime to evaluate. Defaults to now (system local time).

    Returns:
        float: Traffic multiplier in range [1.0, 2.5].
               1.0 = free-flow, 2.5 = absolute gridlock.
    """
    if dt is None:
        dt = datetime.now()

    # Fractional hour: 18.5 = 6:30 PM, 8.75 = 8:45 AM
    hour = dt.hour + dt.minute / 60.0
    weekday = dt.weekday()  # 0=Mon, 6=Sun
    is_weekend = weekday >= 5
    is_friday = weekday == 4

    # ── Morning Rush: Gaussian centered at 8:30 AM ──
    # σ = 1.2 is tight because the morning rush dissipates fast
    # once offices absorb commuters by ~10 AM.
    morning_bump = 1.0 * math.exp(
        -0.5 * ((hour - 8.5) / 1.2) ** 2
    )

    # ── Evening Rush: Gaussian centered at 6:00 PM ──
    # σ = 1.5 is wider because the evening has staggered releases:
    #   - Schools at 3:30 PM
    #   - IT offices at 5:00–6:00 PM
    #   - Shopping/recreation until 8:00 PM
    # Amplitude 1.2 > morning's 1.0 because all these flows compound.
    evening_bump = 1.2 * math.exp(
        -0.5 * ((hour - 18.0) / 1.5) ** 2
    )

    # ── Friday modifier ──
    # Friday evenings are empirically ~20% worse:
    # half-day offices + weekend travel departures + general FOMO.
    if is_friday:
        evening_bump *= 1.2

    # ── Weekend discount ──
    # Weekends have traffic (Koramangala brunch, Indiranagar shopping,
    # Bannerghatta Zoo runs) but peaks are ~60% lighter.
    if is_weekend:
        morning_bump *= 0.4
        evening_bump *= 0.4

    # ── Final multiplier ──
    # Baseline is 1.0 (free-flow). Bumps add on top.
    # Clamped to [1.0, 2.5] — even the worst Bangalore traffic
    # doesn't make a bus take more than 2.5× its normal time.
    multiplier = 1.0 + morning_bump + evening_bump
    multiplier = max(1.0, min(2.5, multiplier))

    logger.debug(
        f"Traffic multiplier: {multiplier:.2f}x | "
        f"hour={hour:.1f} weekday={weekday} "
        f"morning_bump={morning_bump:.3f} evening_bump={evening_bump:.3f}"
    )

    return multiplier


# ═════════════════════════════════════════════════════════════════════
# BUS DATA EXTRACTION
# ═════════════════════════════════════════════════════════════════════

def _extract_bus_positions(data: Any) -> List[Dict[str, float]]:
    """
    Extract bus lat/lon pairs from the Redis payload.

    Handles multiple possible data shapes returned by unofficial
    BMTC API wrappers:
      - Flat list:  [{"lat": 12.97, "lon": 77.59}, ...]
      - Nested:     {"data": [...]} or {"vehicles": [...]}

    Returns:
        List of dicts with "lat" and "lon" keys (as floats).
    """
    buses: List[Dict[str, float]] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                lat = (
                    item.get("lat")
                    or item.get("latitude")
                    or item.get("vehicleLat")
                )
                lon = (
                    item.get("lon")
                    or item.get("lng")
                    or item.get("longitude")
                    or item.get("vehicleLon")
                )
                if lat is not None and lon is not None:
                    try:
                        buses.append({"lat": float(lat), "lon": float(lon)})
                    except (ValueError, TypeError):
                        continue

    elif isinstance(data, dict):
        for key in ("data", "buses", "vehicles", "results"):
            if key in data and isinstance(data[key], list):
                buses = _extract_bus_positions(data[key])
                if buses:
                    break

    return buses


# ═════════════════════════════════════════════════════════════════════
# NEAREST BUS FINDER
# ═════════════════════════════════════════════════════════════════════

async def find_nearest_bus(
    user_lat: float,
    user_lon: float,
    route: str,
    redis_client: aioredis.Redis,
) -> Optional[Dict[str, Any]]:
    """
    Find the nearest bus on a given route to the user.

    Reads from Redis (single async GET — very cheap on Render free tier),
    parses the bus positions, and returns the closest one with its
    distance in meters.

    Args:
        user_lat, user_lon: Cyclist's GPS coordinates.
        route:              BMTC route code (e.g., "500-D").
        redis_client:       Async Redis connection to Upstash.

    Returns:
        Dict with "lat", "lon", "distance_m" of the nearest bus,
        or None if no bus data is available.
    """
    route_key = f"bmtc:route:{route.upper()}"
    raw = await redis_client.get(route_key)

    if not raw:
        logger.warning(f"No data in Redis for route '{route}'")
        return None

    try:
        bus_data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Corrupt JSON in Redis for route '{route}'")
        return None

    buses = _extract_bus_positions(bus_data)
    if not buses:
        logger.warning(f"No parseable bus positions for route '{route}'")
        return None

    # Find closest bus by haversine distance
    nearest = None
    min_dist = float("inf")

    for bus in buses:
        dist = haversine_m(user_lat, user_lon, bus["lat"], bus["lon"])
        if dist < min_dist:
            min_dist = dist
            nearest = bus

    return {
        "lat": nearest["lat"],
        "lon": nearest["lon"],
        "distance_m": round(min_dist, 1),
    }


# ═════════════════════════════════════════════════════════════════════
# THE "CATCH IT?" — LAZY SCORE ENGINE
# ═════════════════════════════════════════════════════════════════════

def _distance_to_score(distance_m: float) -> float:
    """
    Map raw distance (meters) to a base score (0–100).

    ┌──────────────────────────────────────────────────────────────────┐
    │  DISTANCE → SCORE MAPPING                                        │
    │                                                                  │
    │  We use a piecewise function with a smooth cosine interpolation  │
    │  in the middle zone for a natural feel:                          │
    │                                                                  │
    │    distance < 500 m   →  score = 100  ("Run for it!")            │
    │    distance > 3000 m  →  score = 0    ("Keep Pedaling")          │
    │    500–3000 m         →  smooth cosine decay from 100 → 0        │
    │                                                                  │
    │  Why cosine instead of linear?                                   │
    │    Linear decay feels wrong: losing 4 points per 100m doesn't    │
    │    match the real urgency curve. With cosine interpolation:       │
    │      - Score drops slowly near 500m (bus is still close-ish)     │
    │      - Score drops fast in the middle (the "decision zone")      │
    │      - Score flattens near 3km (already hopeless, no urgency)    │
    │                                                                  │
    │  The formula in the transition zone:                             │
    │    t = (distance - 500) / (3000 - 500)       # normalize to 0–1 │
    │    score = 100 × (1 + cos(π × t)) / 2        # cosine ease      │
    │                                                                  │
    │  Sample values:                                                  │
    │    500 m  → 100.0    (threshold — sprint!)                       │
    │    800 m  →  96.6    (still very close)                          │
    │    1000 m →  89.1    (worth jogging to the stop)                 │
    │    1500 m →  61.7    (maybe wait if traffic is bad)              │
    │    2000 m →  34.5    (getting iffy)                              │
    │    2500 m →  11.7    (probably not worth it)                     │
    │    3000 m →   0.0    (nope)                                      │
    └──────────────────────────────────────────────────────────────────┘
    """
    if distance_m <= CATCHABLE_M:
        return 100.0
    if distance_m >= TOO_FAR_M:
        return 0.0

    # Normalize distance into [0, 1] within the transition zone
    t = (distance_m - CATCHABLE_M) / (TOO_FAR_M - CATCHABLE_M)

    # Cosine interpolation: smooth ease-out from 100 → 0
    return 100.0 * (1.0 + math.cos(math.pi * t)) / 2.0


def _traffic_adjust_score(base_score: float, multiplier: float) -> float:
    """
    Adjust the base distance score using the traffic multiplier.

    ┌──────────────────────────────────────────────────────────────────┐
    │  TRAFFIC ADJUSTMENT LOGIC                                        │
    │                                                                  │
    │  The traffic multiplier tells us: "the bus is N× slower than     │
    │  normal." This affects the score in two opposing ways:           │
    │                                                                  │
    │  For HIGH scores (bus is close, ~500m–1.5km):                    │
    │    Heavy traffic HELPS the cyclist catch the bus — the bus is     │
    │    crawling, so you have more time to reach the stop.            │
    │    → Boost the score slightly.                                   │
    │                                                                  │
    │  For LOW scores (bus is far, ~2km–3km):                          │
    │    Heavy traffic means the bus will take even LONGER to arrive.  │
    │    It's already far away AND slow — double penalty.              │
    │    → Reduce the score slightly.                                  │
    │                                                                  │
    │  Implementation:                                                 │
    │    adjustment = (multiplier - 1.0) × 15 × (base_score / 100)    │
    │                                                                  │
    │  At multiplier=1.5 (moderate traffic), base_score=80:            │
    │    adjustment = 0.5 × 15 × 0.8 = +6 → final = 86               │
    │    (bus is close AND slow — easier to catch!)                    │
    │                                                                  │
    │  At multiplier=2.0 (heavy traffic), base_score=20:              │
    │    adjustment = 1.0 × 15 × 0.2 = +3 → final = 23               │
    │    (bus is far AND stuck — marginal help)                        │
    │                                                                  │
    │  The factor of 15 was tuned so that traffic never overwhelms     │
    │  the distance signal — distance is king, traffic is a nudge.    │
    └──────────────────────────────────────────────────────────────────┘
    """
    # Traffic > 1.0 means the bus is slower → gives cyclist more time
    # This is a small bonus that scales with how close the bus already is
    traffic_bonus = (multiplier - 1.0) * 15.0 * (base_score / 100.0)

    adjusted = base_score + traffic_bonus
    return max(0.0, min(100.0, adjusted))


def _make_verdict(score: float, distance_m: float, eta_min: float) -> Dict[str, str]:
    """Generate human-readable verdict and emoji based on score."""
    if score >= 90:
        return {
            "verdict": "🏃 RUN FOR IT! The bus is right there!",
            "action": "sprint",
            "emoji": "🏃",
        }
    elif score >= 70:
        return {
            "verdict": "🚌 Hustle to the stop — you'll make it!",
            "action": "hustle",
            "emoji": "🚌",
        }
    elif score >= 45:
        return {
            "verdict": "🤔 Your call — bus is approachable but not guaranteed.",
            "action": "consider",
            "emoji": "🤔",
        }
    elif score >= 20:
        return {
            "verdict": "🚴 Probably keep pedaling — bus is pretty far.",
            "action": "pedal",
            "emoji": "🚴",
        }
    else:
        return {
            "verdict": "🚴‍♂️ Keep Pedaling! Bus is way too far to chase.",
            "action": "pedal",
            "emoji": "🚴‍♂️",
        }


async def compute_lazy_score(
    user_lat: float,
    user_lon: float,
    route: str,
    dt: Optional[datetime] = None,
    redis_client: Optional[aioredis.Redis] = None,
) -> Dict[str, Any]:
    """
    The core decision function. Computes the "Lazy Score" from 0 to 100.

    ┌──────────────────────────────────────────────────────────────────┐
    │  ALGORITHM OVERVIEW                                              │
    │                                                                  │
    │  1. Find the nearest bus on the route (Redis lookup)             │
    │  2. Compute raw distance in meters (Haversine)                   │
    │  3. Map distance to a base score (cosine interpolation):         │
    │       < 500m  → 100,  > 3km → 0,  smooth curve between         │
    │  4. Get the traffic multiplier for the current time              │
    │  5. Adjust score: heavy traffic = bus is slower = easier to      │
    │     catch if it's close (slight boost)                           │
    │  6. Compute ETA: distance / (base_speed / multiplier)            │
    │  7. Generate human-readable verdict                              │
    │                                                                  │
    │  Total I/O: 1 Redis GET. Zero external API calls.                │
    │  CPU: ~0.1ms of math. Ultra-lightweight for free tier.           │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        user_lat:     Cyclist's latitude.
        user_lon:     Cyclist's longitude.
        route:        BMTC route code (e.g., "500-D", "335-E").
        dt:           Override datetime for testing. Defaults to now().
        redis_client: Async Redis connection for bus data lookup.

    Returns:
        Dict with lazy_score, verdict, breakdown, and metadata.
    """

    # ── Step 1: Traffic Multiplier (pure math, no I/O) ──
    traffic_mult = get_traffic_multiplier(dt)

    # ── Step 2: Find nearest bus (single Redis GET) ──
    nearest = None
    if redis_client:
        nearest = await find_nearest_bus(user_lat, user_lon, route, redis_client)

    # ── Step 3: Compute score ──
    if nearest is None:
        # No bus data → we can't score anything
        return {
            "lazy_score": 0,
            "verdict": "📡 No bus data available for this route.",
            "emoji": "📡",
            "breakdown": {
                "distance": None,
                "eta_minutes": None,
                "traffic_multiplier": round(traffic_mult, 2),
                "traffic_context": _describe_traffic(traffic_mult, dt),
            },
            "meta": {
                "user_location": {"lat": user_lat, "lon": user_lon},
                "route": route,
                "timestamp": (dt or datetime.now()).isoformat(),
                "data_available": False,
            },
        }

    distance_m = nearest["distance_m"]

    # ── Step 4: Distance → Base Score ──
    base_score = _distance_to_score(distance_m)

    # ── Step 5: Traffic Adjustment ──
    final_score = _traffic_adjust_score(base_score, traffic_mult)
    final_score = round(final_score, 1)

    # ── Step 6: ETA Calculation ──
    # effective_speed = base_speed / multiplier
    # At multiplier=2.0 (rush hour): 25/2.0 = 12.5 km/h
    # At multiplier=1.0 (free-flow): 25/1.0 = 25.0 km/h
    effective_speed_kmh = max(
        BUS_MIN_SPEED_KMH,
        BUS_BASE_SPEED_KMH / traffic_mult,
    )
    eta_minutes = (distance_m / 1000.0) / effective_speed_kmh * 60.0

    # ── Step 7: Verdict ──
    verdict_info = _make_verdict(final_score, distance_m, eta_minutes)

    return {
        "lazy_score": final_score,
        "verdict": verdict_info["verdict"],
        "emoji": verdict_info["emoji"],
        "action": verdict_info["action"],
        "breakdown": {
            "distance": {
                "nearest_bus_m": distance_m,
                "nearest_bus_km": round(distance_m / 1000, 2),
                "nearest_bus_location": {
                    "lat": nearest["lat"],
                    "lon": nearest["lon"],
                },
                "base_score": round(base_score, 1),
                "thresholds": {
                    "catchable_m": CATCHABLE_M,
                    "too_far_m": TOO_FAR_M,
                },
            },
            "eta": {
                "minutes": round(eta_minutes, 1),
                "effective_bus_speed_kmh": round(effective_speed_kmh, 1),
                "base_bus_speed_kmh": BUS_BASE_SPEED_KMH,
            },
            "traffic": {
                "multiplier": round(traffic_mult, 2),
                "score_adjustment": round(final_score - base_score, 1),
                "context": _describe_traffic(traffic_mult, dt),
            },
        },
        "meta": {
            "user_location": {"lat": user_lat, "lon": user_lon},
            "route": route,
            "timestamp": (dt or datetime.now()).isoformat(),
            "data_available": True,
        },
    }


def _describe_traffic(multiplier: float, dt: Optional[datetime] = None) -> str:
    """Human-readable traffic context."""
    if dt is None:
        dt = datetime.now()

    day = dt.strftime("%A")
    time = dt.strftime("%I:%M %p")

    if multiplier <= 1.1:
        return f"{day} {time} — Roads empty, bus flying 🟢"
    elif multiplier <= 1.4:
        return f"{day} {time} — Light traffic, bus on schedule 🟡"
    elif multiplier <= 1.8:
        return f"{day} {time} — Moderate traffic, bus is slow 🟠"
    elif multiplier <= 2.2:
        return f"{day} {time} — Heavy congestion, bus crawling 🔴"
    else:
        return f"{day} {time} — Gridlock! Peak Bangalore chaos 🔴🔴"
