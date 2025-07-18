import time

import requests
import json

# === GraphHopper Route Info ===
def get_route_info():
    url = "http://localhost:8989/route"
    params = {
        "point": ["26.371185,-80.102607", "26.373887,-80.108457"],  # [FROM, TO]
        "profile": "car",
        "locale": "en",
        "calc_points": "false",
        "instructions": "false"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "paths" in data:
        path = data["paths"][0]
        eta_minutes = int(path["time"] / 60000)
        distance_km = path["distance"] / 1000
        print(f"🕒 Estimated Arrival Time: {eta_minutes} min")
        print(f"🛣️ Distance: {distance_km:.2f} km")
    else:
        print("❌ Could not fetch route.")

# === HERE Traffic Incidents Info ===
def get_traffic_incidents():
    url = "https://traffic.ls.hereapi.com/traffic/6.3/incidents.json"
    params = {
        "bbox": "26.20,-80.25;26.50,-80.00",  # Adjust as needed
        "criticality": "minor,major",
        "apiKey": "YOUR_HERE_API_KEY"
    }

    response = requests.get(url, params=params)
    incidents = response.json().get("TRAFFIC_ITEMS", {}).get("TRAFFIC_ITEM", [])

    print("\n🚨 Traffic Incidents:")
    if not incidents:
        print("No incidents found.")
    for i, item in enumerate(incidents):
        desc = item["TRAFFIC_ITEM_DESCRIPTION"][0]["value"]
        type_desc = item["TRAFFIC_ITEM_TYPE_DESC"]
        print(f"{i+1}. {type_desc} - {desc}")

# === Run It All ===
if __name__ == "__main__":
    while True:
        get_route_info()
        get_traffic_incidents()
        print("\n--- Updating again in 5 minutes ---\n")
        time.sleep(500)

