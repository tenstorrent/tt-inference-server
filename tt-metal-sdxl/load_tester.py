import requests
import time

API_URL = "http://localhost:8000/image/generations"
AUTH_TOKEN = "your-secret-key"
LOG_FILE = "api_status.log"

# JSON payload you want to send
PAYLOAD = {
    "prompt": "Michael Jordan blocked by Spud Webb",
    "num_inference_step": 5
}

# Headers matching your curl command
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def check_api():
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=PAYLOAD, headers=HEADERS, timeout=30)
        elapsed = time.time() - start_time
        return "ok", elapsed if response.status_code == 200 else "nok"
    except Exception as e:
        return e

def main():
    while True:
        status, elapsed = check_api()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log:
            log.write(f"{timestamp} - {status} time: {elapsed}\n")
        print(f"{timestamp} - {status}")
        time.sleep(15)

if __name__ == "__main__":
    main()
