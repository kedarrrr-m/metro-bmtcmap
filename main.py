import json
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from core.config import settings
from intelligence import compute_lazy_score, haversine_m

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BMTC Live Tracker",
    description="24/7 Live Bangalore Transit Tracker",
    version="1.0.0",
)

# CORS — allow the browser frontend to call the API freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Serve static files (the Leaflet map UI) at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

redis_client = None
route_index = None


@app.on_event("startup")
async def startup_event():
    global redis_client, route_index
    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    
    try:
        with open("core/bmtc_route_index.json", "r", encoding="utf-8") as f:
            route_index = json.load(f).get("stops", [])
            logger.info(f"Loaded {len(route_index)} GTFS stops for routing.")
    except Exception as e:
        logger.error(f"Failed to load route index: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Leaflet map UI at the root URL."""
    return FileResponse("static/index.html")


@app.get("/api/v1/find-route")
async def find_route(
    originLat: float = Query(...),
    originLon: float = Query(...),
    destLat: float = Query(...),
    destLon: float = Query(...)
):
    """
    Finds a direct BMTC bus route between Origin and Destination coordinates
    by analyzing the GTFS spatial index (pre-processed). Max walking distance: 800m.
    """
    if not route_index:
        raise HTTPException(status_code=500, detail="Route index not available")

    # 1. Find routes serving stops near Origin
    origin_routes = set()
    for stop in route_index:
        if haversine_m(originLat, originLon, stop["lat"], stop["lon"]) <= 800:
            origin_routes.update(stop["routes"])

    # 2. Find routes serving stops near Destination
    dest_routes = set()
    for stop in route_index:
        if haversine_m(destLat, destLon, stop["lat"], stop["lon"]) <= 800:
            dest_routes.update(stop["routes"])

    # 3. Intersect
    direct_routes = origin_routes.intersection(dest_routes)
    
    if not direct_routes:
        return {"route": None, "message": "No direct route found within walking distance."}
    
    routes_list = sorted(list(direct_routes))
    
    # Ideally pick a good route, but just returning alphabetical first for now
    return {
        "route": routes_list[0],
        "alternatives": routes_list[1:6]
    }


@app.get("/api/v1/buses/{route}")
async def get_buses_for_route(route: str):
    """
    Fetch the latest bus coordinates for a specific route from Redis.
    Also registers the route as 'active' so the background poller knows
    to fetch data for it.
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis connection not established")

    route_upper = route.upper()
    
    # 1. Register this route as active for the next 10 minutes (600s)
    # The poller.py script looks for keys matching active_route:*
    await redis_client.setex(f"active_route:{route_upper}", 600, "1")

    # 2. Fetch the actual bus coordinates
    route_key = f"bmtc:route:{route_upper}"
    data = await redis_client.get(route_key)

    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"route": route_upper, "data": data}

    return {
        "message": (
            f"No data found for route {route_upper}. "
            "The background worker has been notified and might still be warming up."
        )
    }


import random

@app.get("/api/v1/mock-bmtc")
async def mock_bmtc_api(route: str = Query(...)):
    """
    Fallback mock endpoint for when the unofficial BMTC API is down.
    Generates 3-5 fake buses along the requested route.
    """
    if not route_index:
        return {"data": []}

    # Find stops for this route
    route_stops = [s for s in route_index if route.upper() in s["routes"]]
    if not route_stops:
        # Fallback to IIITB area if route not found
        return {"data": [{"lat": 12.9449 + random.uniform(-0.02, 0.02), "lon": 77.6069 + random.uniform(-0.02, 0.02)} for _ in range(3)]}

    buses = []
    num_buses = random.randint(3, 8)
    
    # Pick random stops and place a bus slightly offset from them
    sampled_stops = random.choices(route_stops, k=num_buses)
    for stop in sampled_stops:
        buses.append({
            "lat": stop["lat"] + random.uniform(-0.005, 0.005),
            "lon": stop["lon"] + random.uniform(-0.005, 0.005),
            "route": route.upper()
        })
        
    return {"data": buses}

@app.get("/api/v1/lazy-score")
async def get_lazy_score(
    lat: float = Query(..., description="Cyclist's latitude"),
    lon: float = Query(..., description="Cyclist's longitude"),
    route: str = Query("500-D", description="BMTC route code (e.g. 500-D, 335-E)"),
):
    """
    The "Catch It?" endpoint.

    Computes a Lazy Score from 0–100 telling a cyclist whether to
    keep pedaling or chase the next BMTC bus, based on:
      - Distance to the nearest bus (Haversine from Redis data)
      - Bangalore Traffic Multiplier (time-of-day Gaussian model)

    Example:
        GET /api/v1/lazy-score?lat=12.9716&lon=77.5946&route=500-D
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis connection not established")

    result = await compute_lazy_score(
        user_lat=lat,
        user_lon=lon,
        route=route.upper(),
        redis_client=redis_client,
    )
    return result
