import os
from dotenv import load_dotenv
import requests

# Load credentials from .env file
load_dotenv()  
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
s = requests.Session()

# Save credentials into session
s.auth = (username, password)

# Send a POST request to the /authentication API
response = s.post('https://api.worldquantbrain.com/authentication')

if response.status_code == 201:
    print("Simulation sent successfully.")
else:
    print("Failed to send simulation.")
    print("Status Code:", response.status_code)
    print("Response:", response.text)

def main():
    # Run tests if requested
    from src.genetic import GPLearnSimulator
    simulator = GPLearnSimulator(session=s)
    simulator.evolve()


if __name__ == "__main__":
    main()
