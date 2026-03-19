# passenger_portal.py
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import requests
import threading
import time
import os

# ---------------- CONFIG ----------------
DRIVER_SERVER = "http://127.0.0.1:5000"  # Driver's Flask server
FETCH_INTERVAL = 1.0  # seconds
BEEP_PATH = os.path.join("static", "beep.mp3")  # Place beep.mp3 in static folder

# ---------------- FLASK + SOCKETIO ----------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

current_driver_status = {
    "driver_id": None,
    "status": "Alert",  # Default to Alert/Normal
    "timestamp": None
}

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    """Render the passenger live status page."""
    return render_template("index.html")

@app.route("/status")
def get_status():
    """Fallback API to get the latest driver status."""
    return jsonify(current_driver_status)

# ---------------- FETCH DRIVER STATUS ----------------
def fetch_driver_status():
    """
    Background thread: fetches the latest driver status from the driver server
    and emits it via SocketIO if it changes.
    """
    global current_driver_status
    last_status = current_driver_status["status"]

    while True:
        try:
            res = requests.get(f"{DRIVER_SERVER}/latest_alert", timeout=2)
            if res.status_code == 200:
                data = res.json()
                new_status = data.get("status", "Alert") or "Alert"  # Default to Alert

                # Normalize status for display
                if new_status.lower() in ["normal", "alert"]:
                    new_status = "Alert"
                    data["status"] = "Alert"

                # Only emit if status changed
                if new_status != last_status:
                    current_driver_status.update(data)
                    socketio.emit("driver_update", current_driver_status)
                    last_status = new_status

        except Exception as e:
            print(f"⚠️ Error fetching driver status: {e}")

        time.sleep(FETCH_INTERVAL)

# ---------------- SOCKETIO EVENTS ----------------
@socketio.on("connect")
def on_connect():
    """Send the current driver status immediately when a passenger connects."""
    print("Passenger connected")
    socketio.emit("driver_update", current_driver_status)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    # Ensure static folder exists for beep
    if not os.path.exists("static"):
        os.makedirs("static")
    if not os.path.exists(BEEP_PATH):
        print(f"⚠️ Place your beep sound at {BEEP_PATH}")

    # Start background thread to fetch driver status
    threading.Thread(target=fetch_driver_status, daemon=True).start()

    # Run Flask + SocketIO
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)
