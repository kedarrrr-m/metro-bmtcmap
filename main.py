import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from core.config import settings
from intelligence import compute_lazy_score

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


@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Leaflet map UI at the root URL."""
    return FileResponse("static/index.html")


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

    return {
        "message": (
            f"No data found for route {route}. "
            "The background worker might still be warming up."
        )
    }


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
