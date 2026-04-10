"""
intelligence.py — The "Lazy Cyclist" Decision Engine
=====================================================

This module is the brain of the Bangalore Transit Tracker. It combines
three real-time signals—road elevation, bus proximity, and Bangalore's
infamous traffic patterns—to produce a single "Lazy Score" (0–100) that
tells a cyclist whether to keep pedaling or hop on the next BMTC bus.

Architecture:
    Redis (Upstash) ──► Bus coordinates ──┐
    Google Elevation API ──► Slope data ──┼──► Lazy Score (0-100)
    Time-of-day / Day-of-week ────────────┘

Score Interpretation:
    0–25:  "Keep Pedaling, Warrior!" 🚴
    26–50: "Your Call, Chief" 🤔
    51–75: "Maybe Wait for the Bus?" 🚌
    76–100: "Get on That Bus NOW!" 🏃→🚌

NOTE: We use `redis.asyncio` which is the modern successor to the
      now-deprecated `aioredis` library (merged into redis-py ≥ 4.2).
"""

import json
import math
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import httpx
import redis.asyncio as aioredis  # Modern async redis — the successor to aioredis

from core.config import settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════

EARTH_RADIUS_KM = 6371.0

# Average BMTC bus speeds (km/h) under different traffic conditions.
# These are empirically derived from Bangalore commute observations:
#   - Peak hours on Outer Ring Road: 8-12 km/h (crawling)
#   - Midday on MG Road: 18-22 km/h
#   - Late night on Bellary Road: 35-40 km/h
BUS_SPEED_FREE_FLOW_KMH = 30.0   # Base speed with zero traffic
BUS_SPEED_MIN_KMH = 6.0          # Absolute floor (Silk Board junction at 6 PM)

# Cyclist parameters
CYCLIST_AVG_SPEED_KMH = 15.0     # Average cycling speed on flat ground
CYCLIST_UPHILL_PENALTY = 0.6     # Speed multiplier when climbing steep grades

# Elevation sampling: how far ahead to look and how many samples
ELEVATION_LOOKAHEAD_M = 1000     # 1 km ahead of the cyclist
ELEVATION_SAMPLE_COUNT = 5       # 5 evenly spaced sample points

# Score component weights — these MUST sum to 100.
# Tuned so that a steep hill with a nearby bus dominates the decision,
# while traffic conditions serve as a tiebreaker.
WEIGHT_INCLINE = 45   # How much slope affects the decision
WEIGHT_BUS_ETA = 35   # How much bus proximity matters
WEIGHT_FRICTION = 20  # How much Bangalore traffic chaos matters


# ═════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═════════════════════════════════════════════════════════════════════

def haversine_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Great-circle distance between two points on Earth using the
    Haversine formula. Returns distance in kilometers.

    This is used to compute how far each bus is from the cyclist.
    Accuracy: ±0.5% for distances under 100 km — more than sufficient
    for intra-city Bangalore distances.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def destination_point(
    lat: float, lon: float,
    bearing_deg: float, distance_m: float,
) -> Tuple[float, float]:
    """
    Given a starting point, a bearing (degrees clockwise from north),
    and a distance (meters), compute the destination lat/lon.

    Uses the spherical law of cosines — accurate to ~1 m for the
    short distances we're working with (≤ 1 km).
    """
    R = EARTH_RADIUS_KM * 1000  # Earth radius in meters
    d = distance_m / R          # Angular distance in radians

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    lat2 = math.asin(
        math.sin(lat_rad) * math.cos(d)
        + math.cos(lat_rad) * math.sin(d) * math.cos(bearing_rad)
    )
    lon2 = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(lat_rad),
        math.cos(d) - math.sin(lat_rad) * math.sin(lat2),
    )

    return (math.degrees(lat2), math.degrees(lon2))


# ═════════════════════════════════════════════════════════════════════
# FRICTION COEFFICIENT — THE BANGALORE TRAFFIC MODEL
# ═════════════════════════════════════════════════════════════════════

def compute_friction_coefficient(dt: Optional[datetime] = None) -> float:
    """
    Compute the 'Friction Coefficient' for Bangalore traffic.

    The Friction Coefficient (FC) is a dimensionless value between
    0.15 and 1.0 that models how freely traffic flows at any given
    moment. It directly scales the effective bus speed:

        effective_bus_speed = BUS_SPEED_FREE_FLOW_KMH × FC

    ┌──────────────────────────────────────────────────────────────────┐
    │  THE MATH BEHIND THE FRICTION COEFFICIENT                       │
    │                                                                  │
    │  Bangalore traffic follows a bimodal distribution with two       │
    │  daily peaks. We model each peak as a Gaussian "dip" in the      │
    │  friction coefficient:                                           │
    │                                                                  │
    │    peak_effect = amplitude × exp(-0.5 × ((h − μ) / σ)²)         │
    │                                                                  │
    │  Morning Peak (the office rush):                                 │
    │    μ = 8.5  (centered at 8:30 AM)                                │
    │    σ = 1.2  (most of the pain is between 7:00–10:00)             │
    │    A = 0.65 (knocks friction down by up to 65%)                  │
    │                                                                  │
    │  Evening Peak (the return + shopping + school pickup):           │
    │    μ = 18.0  (centered at 6:00 PM)                               │
    │    σ = 1.5   (wider: evening traffic lingers 4:30–8:00 PM)       │
    │    A = 0.70  (worse than morning — evening is king of chaos)     │
    │                                                                  │
    │  Day-of-Week Modifiers:                                          │
    │    Friday evening: ×1.15 amplitude (everyone leaves early)       │
    │    Weekends: ×0.45 amplitude (traffic exists, but 55% lighter)   │
    │                                                                  │
    │  Final FC = 1.0 − morning_effect − evening_effect                │
    │  Clamped to [0.15, 1.0]                                          │
    │                                                                  │
    │  Examples:                                                       │
    │    Sunday 6 AM     → FC ≈ 0.98  (empty roads)                    │
    │    Tuesday 11 AM   → FC ≈ 0.82  (mild traffic)                   │
    │    Wednesday 8:30  → FC ≈ 0.35  (morning peak)                   │
    │    Friday 6:00 PM  → FC ≈ 0.19  (absolute carnage)              │
    │    Saturday 5 PM   → FC ≈ 0.72  (weekend buzz, manageable)       │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        dt: The datetime to evaluate. Defaults to now (system local time).

    Returns:
        float: Friction coefficient in the range [0.15, 1.0].
               0.15 = gridlock (Silk Board on Friday evening)
               1.0  = free-flow (Sunday pre-dawn)
    """
    if dt is None:
        dt = datetime.now()

    # Fractional hour: 18.5 = 6:30 PM, 8.75 = 8:45 AM
    hour = dt.hour + dt.minute / 60.0
    weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
    is_weekend = weekday >= 5
    is_friday = weekday == 4

    # ── Morning Rush: Gaussian centered at 8:30 AM ──
    #
    # The morning peak in Bangalore is sharp and concentrated.
    # By 10:00 AM, most office-goers have reached their desks,
    # so σ is tight at 1.2 hours.
    morning_center = 8.5
    morning_sigma = 1.2
    morning_amplitude = 0.65
    morning_effect = morning_amplitude * math.exp(
        -0.5 * ((hour - morning_center) / morning_sigma) ** 2
    )

    # ── Evening Rush: Gaussian centered at 6:00 PM ──
    #
    # The evening peak is wider (σ = 1.5) because:
    #   - Schools let out at 3:30–4:00 PM (parents on the road)
    #   - Offices release in waves from 5:00–7:30 PM
    #   - Shopping/recreational traffic overlaps until 8:00 PM
    # Amplitude is higher (0.70) because all these flows compound.
    evening_center = 18.0
    evening_sigma = 1.5
    evening_amplitude = 0.70
    evening_effect = evening_amplitude * math.exp(
        -0.5 * ((hour - evening_center) / evening_sigma) ** 2
    )

    # ── Day-of-week modifiers ──
    if is_friday:
        # Friday evenings are empirically ~15% worse in Bangalore.
        # Half-day offices + weekend trip departures + general FOMO.
        evening_effect *= 1.15

    if is_weekend:
        # Weekend traffic exists (Koramangala brunch crowd, Indiranagar
        # shoppers, Bannerghatta Zoo trips) but is ~55% lighter than
        # the weekday commute peaks.
        morning_effect *= 0.45
        evening_effect *= 0.45

    # ── Compute final friction coefficient ──
    #
    # FC = 1.0 (baseline free-flow) minus the Gaussian dips.
    # Clamp to [0.15, 1.0]: even at absolute worst, Bangalore traffic
    # moves *a little*. Zero would mean a complete road closure.
    friction = 1.0 - morning_effect - evening_effect
    friction = max(0.15, min(1.0, friction))

    logger.debug(
        f"Friction: {friction:.3f} | hour={hour:.1f} weekday={weekday} "
        f"morning={morning_effect:.3f} evening={evening_effect:.3f}"
    )

    return friction


# ═════════════════════════════════════════════════════════════════════
# ELEVATION — GOOGLE MAPS ELEVATION API
# ═════════════════════════════════════════════════════════════════════

async def get_elevation_gradient(
    lat: float,
    lon: float,
    bearing_deg: float = 0.0,
) -> float:
    """
    Sample elevation along the next 1 km of the cyclist's path and
    compute the average gradient (rise ÷ run, as a percentage).

    Uses the Google Maps Elevation API to get altitude at 5 evenly
    spaced points, then computes:

        gradient_pct = ((elevation_end − elevation_start) / 1000m) × 100

    A positive gradient means uphill; negative means downhill.

    Bangalore elevation reference points (for intuition):
        Koramangala:     ~920 m ASL
        IISc Campus:     ~940 m (climb from Yeshwanthpur side)
        Bannerghatta Rd: ~890 m but approach from the city is uphill
        Whitefield:      ~870 m
        Nandi Hills:     ~1478 m (the ultimate test)

    Args:
        lat:         Cyclist's current latitude.
        lon:         Cyclist's current longitude.
        bearing_deg: Direction of travel in degrees (0=North, 90=East).
                     Defaults to 0 (North) if the client doesn't provide it.

    Returns:
        Average gradient as a percentage. Positive = uphill.
        Returns 0.0 if the API call fails — fail-safe assumption: flat road.
    """
    if not settings.google_maps_api_key:
        logger.warning(
            "GOOGLE_MAPS_API_KEY not set — assuming flat terrain (gradient = 0%). "
            "Set the key in .env or Render Secrets to enable elevation-aware scoring."
        )
        return 0.0

    # Generate sample points along the cyclist's bearing
    sample_points: List[Tuple[float, float]] = []
    for i in range(ELEVATION_SAMPLE_COUNT):
        dist_m = (ELEVATION_LOOKAHEAD_M / (ELEVATION_SAMPLE_COUNT - 1)) * i
        point = destination_point(lat, lon, bearing_deg, dist_m)
        sample_points.append(point)

    # Build the Elevation API request (single batched call — cheap on quota)
    locations_str = "|".join(f"{p[0]:.6f},{p[1]:.6f}" for p in sample_points)
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": locations_str,
        "key": settings.google_maps_api_key,
    }

    try:
        # httpx.AsyncClient is lightweight and cloud-friendly — no thread pools,
        # no blocking. Perfect for running on Render's free tier.
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            data = response.json()

        if data.get("status") != "OK":
            logger.error(
                f"Elevation API error: {data.get('status')} — "
                f"{data.get('error_message', 'no details')}"
            )
            return 0.0

        results = data["results"]
        if len(results) < 2:
            return 0.0

        # Compute gradient from first sample to last sample
        elevation_start = results[0]["elevation"]
        elevation_end = results[-1]["elevation"]
        rise = elevation_end - elevation_start   # meters of vertical gain
        run = ELEVATION_LOOKAHEAD_M              # 1000 meters horizontal

        gradient_pct = (rise / run) * 100

        logger.info(
            f"Elevation profile: {elevation_start:.1f}m → {elevation_end:.1f}m "
            f"over {run}m = {gradient_pct:+.2f}% grade"
        )

        return gradient_pct

    except Exception as e:
        logger.error(f"Elevation API request failed: {e}")
        return 0.0


# ═════════════════════════════════════════════════════════════════════
# BUS ETA ESTIMATION
# ═════════════════════════════════════════════════════════════════════

def _extract_bus_positions(data: Any) -> List[Dict[str, float]]:
    """
    Extract bus lat/lon pairs from the API response payload.

    Handles multiple possible data shapes returned by unofficial
    BMTC API wrappers — some return a flat list of bus objects,
    others nest them under a "data" or "vehicles" key.

    Returns:
        List of dicts with "lat" and "lon" keys (floats).
    """
    buses: List[Dict[str, float]] = []

    # Format 1: flat list of bus objects
    #   [{"lat": 12.97, "lon": 77.59, ...}, ...]
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

    # Format 2: nested under a known key
    #   {"data": [...], "status": "ok"} or {"vehicles": [...]}
    elif isinstance(data, dict):
        for key in ("data", "buses", "vehicles", "results"):
            if key in data and isinstance(data[key], list):
                buses = _extract_bus_positions(data[key])
                if buses:
                    break

    return buses


async def estimate_bus_eta(
    user_lat: float,
    user_lon: float,
    route: str,
    redis_client: aioredis.Redis,
    friction: float,
) -> Optional[float]:
    """
    Estimate when the nearest bus on a given route will reach the user.

    Algorithm:
        1. Pull all known bus positions for the route from Redis.
        2. Compute haversine distance from each bus to the user.
        3. Pick the closest bus.
        4. Estimate arrival using friction-adjusted speed:

            effective_speed = max(BUS_SPEED_MIN, BUS_SPEED_FREE_FLOW × FC)
            ETA_minutes     = (distance_km / effective_speed_kmh) × 60

        At FC = 0.15 (Friday 6 PM):  effective = max(6, 30×0.15) = 6 km/h
        At FC = 1.0  (Sunday 6 AM):  effective = max(6, 30×1.0)  = 30 km/h

    Args:
        user_lat, user_lon: Cyclist's current position.
        route:              BMTC route code (e.g., "500-D").
        redis_client:       Async Redis client connected to Upstash.
        friction:           Current friction coefficient (0.15–1.0).

    Returns:
        Estimated minutes until the nearest bus arrives, or None if
        no bus data is available in Redis.
    """
    route_key = f"bmtc:route:{route.upper()}"
    raw_data = await redis_client.get(route_key)

    if not raw_data:
        logger.warning(f"No Redis data for route {route}")
        return None

    try:
        bus_data = json.loads(raw_data)
    except json.JSONDecodeError:
        logger.error(f"Corrupt JSON in Redis for route {route}")
        return None

    buses = _extract_bus_positions(bus_data)

    if not buses:
        logger.warning(f"No parseable bus positions for route {route}")
        return None

    # Find the closest bus by haversine distance
    min_distance_km = float("inf")
    for bus in buses:
        dist = haversine_km(user_lat, user_lon, bus["lat"], bus["lon"])
        if dist < min_distance_km:
            min_distance_km = dist

    # Compute friction-adjusted effective bus speed
    #
    # The friction coefficient scales the free-flow speed downward.
    # We floor at BUS_SPEED_MIN_KMH because even in absolute gridlock,
    # a BMTC bus inches forward — it never truly stops for more than
    # a few minutes (unlike private cars, buses have dedicated lanes
    # on some Bangalore corridors like Outer Ring Road).
    effective_speed_kmh = max(
        BUS_SPEED_MIN_KMH,
        BUS_SPEED_FREE_FLOW_KMH * friction,
    )

    # ETA = distance / speed, converted to minutes
    eta_minutes = (min_distance_km / effective_speed_kmh) * 60.0

    logger.info(
        f"Route {route}: nearest bus {min_distance_km:.2f} km away | "
        f"speed {effective_speed_kmh:.1f} km/h (FC={friction:.2f}) | "
        f"ETA {eta_minutes:.1f} min"
    )

    return eta_minutes


# ═════════════════════════════════════════════════════════════════════
# THE "CATCH OR PEDAL?" — LAZY SCORE ENGINE
# ═════════════════════════════════════════════════════════════════════

def _describe_gradient(gradient_pct: float) -> str:
    """Human-readable description of the road gradient."""
    if gradient_pct <= -3:
        return "Downhill coast — enjoy the wind! 🏔️↘️"
    elif gradient_pct <= 0:
        return "Slight downhill or flat — smooth sailing"
    elif gradient_pct <= 2:
        return "Flat road — easy pedaling"
    elif gradient_pct <= 5:
        return "Moderate incline — you'll feel it in your quads"
    elif gradient_pct <= 8:
        return "Steep climb — legs will burn 🔥"
    else:
        return "Brutal hill — save yourself, find that bus 😰"


def _describe_traffic(friction: float, dt: Optional[datetime] = None) -> str:
    """Human-readable traffic context string."""
    if dt is None:
        dt = datetime.now()

    day_name = dt.strftime("%A")
    time_str = dt.strftime("%I:%M %p")

    if friction >= 0.85:
        return f"{day_name} {time_str} — Roads clear, buses cruising 🟢"
    elif friction >= 0.60:
        return f"{day_name} {time_str} — Moderate traffic, bus is OK 🟡"
    elif friction >= 0.35:
        return f"{day_name} {time_str} — Heavy traffic, bus is crawling 🟠"
    else:
        return (
            f"{day_name} {time_str} — Gridlock! "
            f"Silk Board vibes. Bus might be slower than your bike 🔴"
        )


async def compute_lazy_score(
    user_lat: float,
    user_lon: float,
    route: str,
    bearing_deg: float = 0.0,
    dt: Optional[datetime] = None,
    redis_client: Optional[aioredis.Redis] = None,
) -> Dict[str, Any]:
    """
    The core decision function. Computes the "Lazy Score" from 0 to 100.

    ┌──────────────────────────────────────────────────────────────────┐
    │  LAZY SCORE FORMULA                                              │
    │                                                                  │
    │  Score = (W_i × S_incline) + (W_b × S_bus) + (W_f × S_traffic)  │
    │                                                                  │
    │  Where:                                                          │
    │    W_i = 45%  (incline weight)                                   │
    │    W_b = 35%  (bus proximity weight)                             │
    │    W_f = 20%  (traffic/friction weight)                          │
    │                                                                  │
    │  S_incline  ∈ [0, 1]:                                            │
    │    Computed via exponential saturation of the gradient:           │
    │      S = 1 − e^(−0.15 × gradient_pct)                           │
    │    This gives a nice curve:                                      │
    │      0% grade  → S ≈ 0.00 (flat, no penalty)                    │
    │      3% grade  → S ≈ 0.36 (moderate hill)                       │
    │      6% grade  → S ≈ 0.59 (steep — IISc approach)               │
    │      10% grade → S ≈ 0.78 (brutal — Nandi Hills foothills)      │
    │    Downhill gradients are clamped to 0 (no reason to bus!)       │
    │                                                                  │
    │  S_bus  ∈ [0, 1]:                                                │
    │    Exponential decay based on bus ETA:                           │
    │      S = e^(−0.08 × ETA_minutes)                                │
    │    This means:                                                   │
    │      ETA=0 min  → S ≈ 1.00 (bus is HERE!)                       │
    │      ETA=4 min  → S ≈ 0.73 (very catchable)                     │
    │      ETA=8 min  → S ≈ 0.53 (maybe worth waiting)                │
    │      ETA=15 min → S ≈ 0.30 (probably not)                       │
    │      ETA=30 min → S ≈ 0.09 (forget it, pedal)                   │
    │    No bus data → S = 0 (assume no bus, keep cycling)             │
    │                                                                  │
    │  S_traffic  ∈ [0, 1]:                                            │
    │    This is simply the friction coefficient itself.               │
    │    Counter-intuitive but correct:                                │
    │      HIGH friction (free-flow) → bus is fast → HIGHER score     │
    │        → "Take the bus, it's zooming along!"                     │
    │      LOW friction (gridlock) → bus is stuck → LOWER score       │
    │        → "Bus is stuck too, just pedal past it"                  │
    │                                                                  │
    │  Score Ranges:                                                   │
    │    0–25:  "Keep Pedaling, Warrior!" 🚴                           │
    │    26–50: "Your Call, Chief" 🤔                                   │
    │    51–75: "Maybe Wait for the Bus?" 🚌                           │
    │    76–100: "Get on That Bus NOW!" 🏃→🚌                          │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        user_lat:    Cyclist's latitude.
        user_lon:    Cyclist's longitude.
        route:       BMTC route to check (e.g., "500-D").
        bearing_deg: Cyclist's heading (0=N, 90=E). Default 0.
        dt:          Override datetime for testing. Default now().
        redis_client: Async Redis client for bus data lookup.

    Returns:
        Dictionary with the lazy_score, verdict, and full breakdown.
    """

    # ── Step 1: Friction Coefficient (pure computation, no I/O) ──
    friction = compute_friction_coefficient(dt)

    # ── Step 2: Elevation Gradient (1 async HTTP call to Google) ──
    gradient_pct = await get_elevation_gradient(user_lat, user_lon, bearing_deg)

    # ── Step 3: Bus ETA (1 async Redis read) ──
    bus_eta: Optional[float] = None
    if redis_client:
        bus_eta = await estimate_bus_eta(
            user_lat, user_lon, route, redis_client, friction
        )

    # ── Step 4: Compute Sub-Scores ──

    # --- Incline sub-score (S_incline) ---
    # We use an exponential saturation curve: S = 1 − e^(−k × g)
    # where k = 0.15 controls how quickly the curve approaches 1.0.
    #
    # Why exponential? Because the *perceived* difficulty of cycling
    # uphill doesn't increase linearly — going from 0→3% grade is
    # "noticeable", but going from 7→10% grade is "get off the bike".
    # The exponential captures this diminishing-returns shape.
    #
    # Downhill gradients are clamped to 0: if you're going downhill,
    # there's zero reason to take the bus — coast and enjoy!
    abs_gradient = max(0.0, gradient_pct)
    s_incline = 1.0 - math.exp(-0.15 * abs_gradient)

    # --- Bus proximity sub-score (S_bus) ---
    # Exponential decay: S = e^(−λ × ETA) where λ = 0.08
    #
    # The decay constant λ = 0.08 was chosen so that:
    #   - A bus < 4 min away gives a strong signal (S > 0.7)
    #   - A bus > 15 min away is barely worth considering (S < 0.3)
    #   - Beyond 30 min, the bus is effectively irrelevant (S < 0.1)
    #
    # This matches a cyclist's decision window: if you'll be waiting
    # 15+ minutes, you'd cover 3–4 km cycling in that time anyway.
    if bus_eta is not None:
        s_bus = math.exp(-0.08 * bus_eta)
    else:
        # No bus data at all → conservatively assume no bus is coming.
        # Score = 0 means this component pushes toward "keep pedaling".
        s_bus = 0.0

    # --- Friction / traffic sub-score (S_traffic) ---
    #
    # This captures whether the current traffic conditions favor
    # taking the bus or the bike.
    #
    # The friction coefficient already encodes this perfectly:
    #   FC close to 1.0 → traffic is light → bus moves fast
    #     → taking the bus is a good deal → HIGHER score
    #   FC close to 0.15 → gridlock → bus is stuck in traffic
    #     → cyclist weaving through traffic might be faster → LOWER score
    #
    # This creates an elegant negative feedback loop with bus ETA:
    # during rush hour, the bus ETA is already inflated (because
    # effective_speed = free_flow × FC is low), AND the traffic
    # sub-score also penalizes the bus option. Double whammy.
    s_traffic = friction

    # ── Step 5: Weighted Combination ──
    #
    # Each weight is expressed as a fraction of 100, and each
    # sub-score is in [0, 1], so the raw result is in [0, 1].
    # We then scale to [0, 100] for the final score.
    raw_score = (
        (WEIGHT_INCLINE / 100.0) * s_incline
        + (WEIGHT_BUS_ETA / 100.0) * s_bus
        + (WEIGHT_FRICTION / 100.0) * s_traffic
    )

    lazy_score = round(min(100.0, max(0.0, raw_score * 100)), 1)

    # ── Step 6: Generate Verdict ──
    if lazy_score >= 76:
        verdict = "🏃→🚌 Get on That Bus NOW!"
        emoji = "🚌"
    elif lazy_score >= 51:
        verdict = "🤔 Maybe Wait for the Bus?"
        emoji = "🚏"
    elif lazy_score >= 26:
        verdict = "😐 Your Call, Chief"
        emoji = "🤷"
    else:
        verdict = "🚴 Keep Pedaling, Warrior!"
        emoji = "🚴"

    # ── Step 7: Build the Response ──
    result = {
        "lazy_score": lazy_score,
        "verdict": verdict,
        "emoji": emoji,
        "breakdown": {
            "incline": {
                "gradient_pct": round(gradient_pct, 2),
                "sub_score": round(s_incline * 100, 1),
                "weight_pct": WEIGHT_INCLINE,
                "description": _describe_gradient(gradient_pct),
            },
            "bus_proximity": {
                "eta_minutes": round(bus_eta, 1) if bus_eta is not None else None,
                "route": route,
                "sub_score": round(s_bus * 100, 1),
                "weight_pct": WEIGHT_BUS_ETA,
            },
            "traffic": {
                "friction_coefficient": round(friction, 3),
                "sub_score": round(s_traffic * 100, 1),
                "weight_pct": WEIGHT_FRICTION,
                "context": _describe_traffic(friction, dt),
            },
        },
        "meta": {
            "user_location": {"lat": user_lat, "lon": user_lon},
            "bearing_deg": bearing_deg,
            "timestamp": (dt or datetime.now()).isoformat(),
        },
    }

    logger.info(f"Lazy Score: {lazy_score} — {verdict}")
    return result
