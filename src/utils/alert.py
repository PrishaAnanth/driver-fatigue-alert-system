import pygame
import requests

# Initialize pygame for sound
pygame.mixer.init()
alert_sound = "assets/beep.wav"

# Flask Passenger Alert Server (update IP/port if hosted elsewhere)
ALERT_SERVER = "http://localhost:5000/alert"


def play_alert():
    """Play local beep alert for driver."""
    try:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
    except Exception as e:
        print("⚠️ Failed to play alert sound:", e)


def send_passenger_alert(message="🚨 Driver is drowsy! Please stay alert."):
    """Send alert message to all connected passengers via Flask server."""
    try:
        requests.post(ALERT_SERVER, json={"msg": message}, timeout=2)
        print(f"✅ Passenger alert sent: {message}")
    except Exception as e:
        print("⚠️ Failed to send passenger alert:", e)


def trigger_full_alert():
    """Play sound locally and notify passengers at the same time."""
    play_alert()
    send_passenger_alert()
