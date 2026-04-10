import os
import csv
import json
import io
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

GTFS_URL = "https://raw.githubusercontent.com/Vonter/bmtc-gtfs/main/gtfs/bmtc.zip"
OUTPUT_FILE = "core/bmtc_route_index.json"

def build():
    logging.info(f"Downloading GTFS from {GTFS_URL}...")
    response = urllib.request.urlopen(GTFS_URL)
    zip_data = response.read()
    logging.info(f"Downloaded {len(zip_data) / 1024 / 1024:.1f} MB. Extracting...")
    
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        # Load routes
        logging.info("Parsing routes.txt...")
        routes = {}
        with z.open("routes.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, 'utf-8'))
            for row in reader:
                # BMTC route_id is often a long string, route_short_name is like "500-D"
                # Sometimes the API expects route_short_name, sometimes long name. We'll track both.
                short_name = row.get("route_short_name", "").strip() or row.get("route_id", "")
                routes[row["route_id"]] = short_name

        # Load trips
        logging.info("Parsing trips.txt...")
        trip_to_route = {}
        with z.open("trips.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, 'utf-8'))
            for row in reader:
                # We just map trip_id to route's short name
                route_id = row.get("route_id")
                trip_to_route[row["trip_id"]] = routes.get(route_id, route_id)

        # Load stop_times and map stop_ids to routes
        logging.info("Parsing stop_times.txt...")
        stop_routes = {}
        with z.open("stop_times.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, 'utf-8'))
            count = 0
            for row in reader:
                stop = row["stop_id"]
                trip = row["trip_id"]
                route = trip_to_route.get(trip)
                if route:
                    if stop not in stop_routes:
                        stop_routes[stop] = set()
                    stop_routes[stop].add(route)
                count += 1
                if count % 1000000 == 0:
                    logging.info(f"  Processed {count/1000000:.1f}M stop_times...")

        # Load stops
        logging.info("Parsing stops.txt...")
        final_stops = []
        with z.open("stops.txt") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, 'utf-8'))
            for row in reader:
                stop_id = row["stop_id"]
                if stop_id in stop_routes:
                    final_stops.append({
                        "stop_id": stop_id,
                        "lat": float(row["stop_lat"]),
                        "lon": float(row["stop_lon"]),
                        "name": row.get("stop_name", stop_id),
                        "routes": list(stop_routes[stop_id])
                    })

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as out:
        json.dump({"stops": final_stops}, out, separators=(',', ':'))
        
    logging.info(f"Saved {len(final_stops)} stops to {OUTPUT_FILE}.")
    logging.info(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    build()
