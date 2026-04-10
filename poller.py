import time
import json
import logging
import requests
import redis
from core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

ROUTES_TO_TRACK = ["500-D", "335-E"]

def fetch_buses(route: str):
    """
    Fetch bus coordinates from the unofficial BMTC API.
    Since endpoints may vary, we pass the route as a query parameter or path.
    A standard placeholder pattern is used here.
    """
    try:
        url = f"{settings.bmtc_api_url}?route={route}"
        response = requests.get(url, timeout=10)
        
        # Some unofficial APIs might return HTML if they are down or limit connections
        if response.status_code == 200:
            try:
                data = response.json()
                return data
            except ValueError:
                logger.error(f"Failed to decode JSON for route {route}. Status: {response.status_code}")
                return None
        else:
            logger.error(f"API Error for route {route}. Status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception for route {route}. Error: {e}")
        return None

def main():
    logger.info("Starting BMTC Data Poller Worker...")
    logger.info(f"Connecting to Redis at: {settings.redis_url.split('@')[-1] if '@' in settings.redis_url else 'local'}")
    
    redis_client = redis.from_url(
        settings.redis_url, 
        encoding="utf-8", 
        decode_responses=True
    )
    
    while True:
        for route in ROUTES_TO_TRACK:
            logger.info(f"Fetching data for route: {route}")
            data = fetch_buses(route)
            
            if data:
                # Store it in Redis with an expiration buffer
                route_key = f"bmtc:route:{route}"
                # Expiry of 2 mins is set. Next poll occurs at 30 seconds.
                redis_client.setex(route_key, 120, json.dumps(data))
                logger.info(f"Updated Redis for route {route}.")
            
        logger.info(f"Waiting {settings.poll_interval_seconds} seconds before next poll...")
        time.sleep(settings.poll_interval_seconds)

if __name__ == "__main__":
    main()
