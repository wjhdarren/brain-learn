import os
from dotenv import load_dotenv
import requests
import sys
from src.function import *
from time import time
import dill

def create_session():
    """Create and authenticate a new session."""
    # Load credentials from .env file
    load_dotenv()
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    
    if not username or not password:
        print("Error: USERNAME or PASSWORD environment variables not set.")
        print("Please check your .env file.")
        sys.exit(1)
    
    # Create a new session
    s = requests.Session()
    s.auth = (username, password)
    
    # Send a POST request to the /authentication API
    response = s.post('https://api.worldquantbrain.com/authentication')
    
    if response.status_code == 201:
        print("Authentication successful.")
        return s
    else:
        print("Failed to authenticate.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return None

def main():
    # Create and authenticate the session
    s = create_session()
    if not s:
        print("Exiting due to authentication failure.")
        return
    
    INIT_POP_LIST = dill.load(open('initial-population.pkl', 'rb'))
    
    # Run the GPLearn simulator
    from src.genetic import GPLearnSimulator
    simulator = GPLearnSimulator(
        session=s,
        population_size = 100,
        generations = 100,
        tournament_size = 3,
        p_crossover = 0.6,
        p_mutation = 0.15,
        p_subtree_mutation = 0.1,
        parsimony_coefficient = 0.02,
        random_state = int(time()/1000),
        #init_population = INIT_POP_LIST
        max_depth = 5,
        max_operators = 6,
        )
    simulator.evolve()

if __name__ == "__main__":
    main()
