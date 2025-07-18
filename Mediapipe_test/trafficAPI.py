import requests


def get_route_info():
    url = "http://localhost:8989/route"
    params = {
        "point": ["26.371185,-80.102607", "26.373887,-80.108457"],
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
        return [
            f"🕒 ETA: {eta_minutes} min",
            f"🛣️ Distance: {distance_km:.2f} km"
        ]
    else:
        return ["❌ Could not fetch route."]


def get_traffic_incidents():
    url = "https://traffic.ls.hereapi.com/traffic/6.3/incidents.json"
    params = {
        "bbox": "26.20,-80.25;26.50,-80.00",
        "criticality": "minor,major",
        "apiKey": "rNdZgY1udnuAr1U3TBnPR3Pp2z-ETma-jgidHDOlDYo"
    }

    response = requests.get(url, params=params)
    incidents = response.json().get("TRAFFIC_ITEMS", {}).get("TRAFFIC_ITEM", [])

    lines = ["🚨 Traffic Incidents:"]
    if not incidents:
        lines.append("No incidents found.")
    else:
        for i, item in enumerate(incidents[:3]):  # limit to 3
            desc = item["TRAFFIC_ITEM_DESCRIPTION"][0]["value"]
            type_desc = item["TRAFFIC_ITEM_TYPE_DESC"]
            lines.append(f"{i+1}. {type_desc}: {desc}")
    return lines


def get_traffic_info_combined():
    return get_route_info() + [""] + get_traffic_incidents()
