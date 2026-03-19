from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# -------------------------------
# Initialize Flask App
# -------------------------------
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (for Streamlit frontend)

# -------------------------------
# In-Memory "Database"
# -------------------------------
alerts_log = []      # Stores recent alerts (max 10)
subscribers = {}     # Placeholder for future real-time notifications (WebSockets)

# -------------------------------
# Route: Driver → POST alert
# -------------------------------
@app.route("/alert", methods=["POST"])
def receive_alert():
    """
    Receives alert messages from the driver app.
    Example POST data:
    {
        "driver_id": "DL123",
        "status": "Drowsy",
        "timestamp": "2025-10-15T12:45:00"
    }
    """
    data = request.json
    driver_id = data.get("driver_id")
    status = data.get("status")
    timestamp = data.get("timestamp") or datetime.now().isoformat()

    if not driver_id or not status:
        return jsonify({"error": "driver_id or status missing"}), 400

    # Construct alert message
    alert_msg = {
        "driver_id": driver_id,
        "status": status,
        "timestamp": timestamp,
        "message": f"⚠️ Driver {driver_id} is {status} at {datetime.now().strftime('%H:%M:%S')}"
    }

    # Log alert (keep only last 10)
    alerts_log.append(alert_msg)
    if len(alerts_log) > 10:
        alerts_log.pop(0)

    # Print to terminal for debugging
    print(alert_msg["message"])

    # Broadcast to subscribers (future WebSocket support)
    if driver_id in subscribers:
        for callback in subscribers[driver_id]:
            callback(alert_msg["message"])

    return jsonify({"success": True, "msg": alert_msg["message"]})

# -------------------------------
# Route: Passenger → GET all alerts
# -------------------------------
@app.route("/alerts", methods=["GET"])
def get_alerts():
    """
    Returns the most recent driver alerts.
    Example: GET http://localhost:5000/alerts
    """
    return jsonify({"alerts": alerts_log})

# -------------------------------
# Route: Latest Alert (single most recent)
# -------------------------------
@app.route("/latest_alert", methods=["GET"])
def get_latest_alert():
    """
    Returns only the most recent alert.
    """
    if alerts_log:
        return jsonify(alerts_log[-1])
    else:
        return jsonify({"driver_id": None, "status": None, "timestamp": None, "message": ""})

# -------------------------------
# Route: Health Check
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚗 Driver Alert Server Running",
        "endpoints": {
            "POST /alert": "Receive alerts from driver app",
            "GET /alerts": "View last 10 driver alerts",
            "GET /latest_alert": "View the most recent alert"
        },
        "total_alerts": len(alerts_log)
    })

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask Alert Server...")
    print("✅ Listening on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
