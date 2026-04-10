import json
from fastapi import FastAPI, HTTPException, Query
import redis.asyncio as redis
from core.config import settings
from intelligence import compute_lazy_score

app = FastAPI(
    title="BMTC Live Tracker",
    description="24/7 Live Bangalore Transit Tracker",
    version="1.0.0"
)

redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    # Connect to Upstash Redis or Local Redis
    redis_client = redis.from_url(        settings.redis_url, 
        encoding="utf-8", 
        decode_responses=True
    )

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "BMTC Transit Tracker is running"}

@app.get("/api/v1/buses/{route}")
async def get_buses_for_route(route: str):
    """
    Fetch the latest bus coordinates for a specific route from Redis.
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis connection not established")
    
    route_key = f"bmtc:route:{route.upper()}"
    data = await redis_client.get(route_key)
    
    if data:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"route": route, "data": data}
    
    return {"message": f"No data found for route {route}. The background worker might still be fetching data or the route is invalid."}

@app.get("/api/v1/lazy-score")
async def get_lazy_score(
    lat: float = Query(..., description="Cyclist's latitude"),
    lon: float = Query(..., description="Cyclist's longitude"),
    route: str = Query("500-D", description="BMTC route code (e.g. 500-D, 335-E)"),
    bearing: float = Query(0.0, description="Direction of travel in degrees (0=North, 90=East)"),
):
    """
    The "Catch or Pedal?" endpoint.

    Computes a Lazy Score from 0–100 telling a cyclist whether to
    keep pedaling or wait for the next BMTC bus, based on:
      - Road elevation ahead (Google Maps Elevation API)
      - Nearest bus ETA on the requested route (Upstash Redis)
      - Current Bangalore traffic conditions (Friction Coefficient)

    Example:
        GET /api/v1/lazy-score?lat=12.9716&lon=77.5946&route=500-D&bearing=45
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis connection not established")

    result = await compute_lazy_score(
        user_lat=lat,
        user_lon=lon,
        route=route.upper(),
        bearing_deg=bearing,
        redis_client=redis_client,
    )
    return result

