from app import app

def test_health_route():
    with app.test_client() as client:
        response = client.get('/health')
        print(f"Status Code: {response.status_code}")
        print(f"Response Data: {response.data.decode('utf-8')}")
        
        if response.status_code == 200 and response.data.decode('utf-8') == "app is live":
            print("✅ /health route verification SUCCESS")
        else:
            print("❌ /health route verification FAILED")

if __name__ == "__main__":
    test_health_route()
