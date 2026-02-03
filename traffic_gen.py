import requests
import time
import random
import sys

BASE_URL = "http://localhost:8000"

def get_analytics():
    try:
        requests.get(f"{BASE_URL}/api/neo/advanced-analytics?days=30")
        print(".", end="", flush=True)
    except Exception as e:
        print("E", end="", flush=True)

def post_prediction():
    data = {
        "absolute_magnitude": random.uniform(15.0, 25.0),
        "estimated_diameter_min": random.uniform(0.1, 1.0),
        "estimated_diameter_max": random.uniform(1.0, 2.0),
        "relative_velocity": random.uniform(10000, 50000),
        "miss_distance": random.uniform(100000, 1000000)
    }
    try:
        requests.post(f"{BASE_URL}/api/neo/predict", json=data)
        print("*", end="", flush=True)
    except Exception as e:
        print("E", end="", flush=True)

def main():
    print(f"Generating traffic to {BASE_URL}...")
    print("Legend: . = Analytics (GET), * = Prediction (POST), E = Error")
    
    start_time = time.time()
    # Run for 30 seconds
    while time.time() - start_time < 30:
        action = random.choice(['analytics', 'predict', 'predict', 'predict'])
        if action == 'analytics':
            get_analytics()
        else:
            post_prediction()
        time.sleep(random.uniform(0.1, 0.5))
    
    print("\nDone! Check your Grafana dashboard now.")

if __name__ == "__main__":
    main()
