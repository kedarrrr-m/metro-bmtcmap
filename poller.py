import json
import logging
import requests
import redis
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def fetch_buses(route: str):
    """
    Fetch bus coordinates from the unofficial BMTC API.
    Returns parsed JSON or None on failure.
    """
    try:
        url = f"{settings.bmtc_api_url}?route={route}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                logger.error(f"Failed to decode JSON for route {route}")
                return None
        else:
            logger.error(f"API error for route {route}: HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for route {route}: {e}")
        return None


def run_once():
    """
    Fetch dynamically requested routes once and push results to Upstash Redis.
    """
    logger.info("BMTC Poller — single run starting")
    logger.info(f"Redis: {settings.redis_url.split('@')[-1] if '@' in settings.redis_url else 'local'}")

    redis_client = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )

    # Find which routes the web backend has marked as active in the last 10 minutes
    active_keys = redis_client.keys("active_route:*")
    
    if not active_keys:
        logger.info("No active routes requested currently. Poller will exit to save quota.")
        return

    routes_to_track = [key.replace("active_route:", "") for key in active_keys]
    logger.info(f"Active routes to poll: {routes_to_track}")

    for route in routes_to_track:
        logger.info(f"Fetching route: {route}")
        data = fetch_buses(route)

        if data:
            # Store with 10-minute expiry
            route_key = f"bmtc:route:{route}"
            redis_client.setex(route_key, 600, json.dumps(data))
            logger.info(f"  ✅ Stored {route} in Redis")
        else:
            logger.warning(f"  ⚠️  No data returned for {route}")

    logger.info("BMTC Poller — run complete")


if __name__ == "__main__":
    run_once()

