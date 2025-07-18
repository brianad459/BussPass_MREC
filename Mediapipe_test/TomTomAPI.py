import cv2
import numpy as np
import requests
import time

# Your TomTom API key
API_KEY = "fkRB1PdWhx9BBuZaajFLm1bIqkH2azQD"

# Target location (Boca Raton)
lat, lon = 26.3665, -80.1289
zoom = 14

# Corrected base URL
base_url = "https://api.tomtom.com/map/1/tile/basic/main"
traffic_url = "https://api.tomtom.com/traffic/4/flowTile/{z}/{x}/{y}/png"


# Convert lat/lon to tile coordinates
def latlon_to_tile(lat, lon, z):
    n = 2 ** z
    xt = int((lon + 180) / 360 * n)
    yt = int((1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * n)
    return xt, yt


# Download and decode tile image
def get_tile(url):
    print("Fetching:", url)
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print("❌ Failed to fetch image. Status:", r.status_code)
            print("Response text:", r.text)
            return None
        if not r.content:
            print("❌ Empty content received.")
            return None
        return cv2.imdecode(np.asarray(bytearray(r.content)), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print("❌ Exception occurred:", e)
        return None


# Get tile coordinates
x, y = latlon_to_tile(lat, lon, zoom)

# Load base map tile
base = get_tile(f"{base_url}/{zoom}/{x}/{y}.png?key={API_KEY}")
if base is None:
    exit("🚫 Base tile could not be loaded. Exiting...")

# Main display loop
while True:
    traffic = get_tile(traffic_url.format(z=zoom, x=x, y=y) + f"?key={API_KEY}")
    if traffic is not None:
        overlay = cv2.addWeighted(base, 1.0, traffic, 0.6, 0)
        print("✅ Traffic overlay applied.")
    else:
        print("⚠️ Skipping traffic overlay due to fetch error.")
        overlay = base.copy()

    if overlay is not None:
        cv2.imshow("Live TomTom Traffic Map", overlay)
    else:
        print("⚠️ Nothing to display.")

    # Refresh every 60 seconds or exit if Esc key is pressed
    if cv2.waitKey(60_000) == 27:
        break

cv2.destroyAllWindows()
