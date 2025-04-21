import requests
from time import sleep
import pandas as pd
from datetime import datetime

def get_alpha_performance(s : requests.Session, alpha_id : str):
    alpha = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id)
    regular = alpha.json().get('regular', {})
    investment_summary = alpha.json().get('is', {})
    checks = alpha.json().get('is', {}).get('checks', [])
    check_results = {check['name']: check['result'] for check in checks}

    # 创建一个包含所需信息的字典
    alpha_performance = {
        'alpha_id' : alpha_id,
        'regular_code': regular.get('code'),
        'turnover': investment_summary.get('turnover'),
        'returns': investment_summary.get('returns'),
        'drawdown': investment_summary.get('drawdown'),
        'margin': investment_summary.get('margin'),
        'fitness': investment_summary.get('fitness'),
        'sharpe': investment_summary.get('sharpe'),
        'LOW_SHARPE': check_results.get('LOW_SHARPE', 'Not Found'),
        'LOW_FITNESS': check_results.get('LOW_FITNESS', 'Not Found'),
        'LOW_TURNOVER': check_results.get('LOW_TURNOVER', 'Not Found'),
        'HIGH_TURNOVER': check_results.get('HIGH_TURNOVER', 'Not Found'),
        'CONCENTRATED_WEIGHT': check_results.get('CONCENTRATED_WEIGHT', 'Not Found'),
        'LOW_SUB_UNIVERSE_SHARPE': check_results.get('LOW_SUB_UNIVERSE_SHARPE', 'Not Found'),
        'SELF_CORRELATION': check_results.get('SELF_CORRELATION', 'Not Found'),
        'MATCHES_COMPETITION': check_results.get('MATCHES_COMPETITION', 'Not Found') }
    return alpha_performance


def simulate(s : requests.Session, fast_expr : str, timeout = 300) -> dict | None:
    simulation_data = {
    'type': 'REGULAR',
    'settings': {
        'instrumentType': 'EQUITY',
        'region': 'USA',
        'universe': 'TOP3000',
        'delay': 1,
        'decay': 1,
        'neutralization': 'INDUSTRY',
        'truncation': 0.1,
        'pasteurization': 'ON',
        'unitHandling': 'VERIFY',
        'nanHandling': 'OFF',
        'language': 'FASTEXPR',
        'visualization': False,
    },
    'regular': fast_expr }
    
    # Maximum number of retries for rate limiting
    MAX_RETRIES = 5
    
    for retry in range(MAX_RETRIES):
        simulation_response = s.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
        
        # Check for rate limiting (429 status code)
        if simulation_response.status_code == 429:
            error_message = simulation_response.text
            if "SIMULATION_LIMIT_EXCEEDED" in error_message:
                # Calculate exponential backoff wait time
                wait_time = 2 ** retry  # 1, 2, 4, 8, 16 seconds
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Rate limit exceeded. Waiting {wait_time} seconds before retry {retry+1}/{MAX_RETRIES}...")
                sleep(wait_time)
                continue  # Retry after waiting
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to send simulation. Status code:", simulation_response.status_code)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Response:", simulation_response.text)
                return None
        
        # Check for authentication failures (401 status code)
        if simulation_response.status_code == 401:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Authentication error: Incorrect credentials.")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Response:", simulation_response.text)
            return None
        
        # If we get here, we have a valid response
        break
    
    # If we've exhausted all retries, return None
    if simulation_response.status_code != 201:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Failed to send simulation after {MAX_RETRIES} retries. Status code:", simulation_response.status_code)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Response:", simulation_response.text)
        return None
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Simulation sent successfully.")
    simulation_progress_url = simulation_response.headers['Location']
    finished = False
    total_wait_time = 0
    
    while not finished and total_wait_time < timeout:
        simulation_progress = s.get(simulation_progress_url)
        
        # Check for authentication failures during progress monitoring
        if simulation_progress.status_code == 401:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Authentication error during simulation progress monitoring.")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Response:", simulation_progress.text)
            return None
        
        # Check if simulation is complete
        if simulation_progress.headers.get("Retry-After", 0) == 0:
            finished = True
            break
            
        # Calculate wait time from header
        wait_time = float(simulation_progress.headers["Retry-After"])
        
        # Update total wait time
        total_wait_time += wait_time
        
        # Check if next wait would exceed timeout
        if total_wait_time >= timeout:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Timeout of {timeout} seconds will be exceeded. Aborting.")
            return None
            
        sleep(wait_time)
        
    if finished:
        try:
            alpha_id = simulation_progress.json()["alpha"] 
            alpha_performance = get_alpha_performance(s, alpha_id)
            # Check if we got a valid result
            if alpha_performance:
                return alpha_performance
            return None
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error processing completed simulation: {e}")
            return None
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Simulation timed out after {total_wait_time} seconds")
        return None
    
def get_alpha_history(s : requests.Session, pandas = True):
    all_alphas = s.get("https://api.worldquantbrain.com/users/self/alphas").json()['results']
    alpha_list = []
    for alpha in all_alphas:
        regular = alpha.get('regular', {})
        investment_summary = alpha.get('is', {})
        checks = alpha.get('is', {}).get('checks', [])

        check_results = {check['name']: check['result'] for check in checks}
        data = {
            'id': alpha.get('id'),
            'regular_code': regular.get('code'),
            'turnover': investment_summary.get('turnover'),
            'returns': investment_summary.get('returns'),
            'drawdown': investment_summary.get('drawdown'),
            'margin': investment_summary.get('margin'),
            'fitness': investment_summary.get('fitness'),
            'sharpe': investment_summary.get('sharpe'),
            'LOW_SHARPE': check_results.get('LOW_SHARPE', 'Not Found'),
            'LOW_FITNESS': check_results.get('LOW_FITNESS', 'Not Found'),
            'LOW_TURNOVER': check_results.get('LOW_TURNOVER', 'Not Found'),
            'HIGH_TURNOVER': check_results.get('HIGH_TURNOVER', 'Not Found'),
            'CONCENTRATED_WEIGHT': check_results.get('CONCENTRATED_WEIGHT', 'Not Found'),
            'LOW_SUB_UNIVERSE_SHARPE': check_results.get('LOW_SUB_UNIVERSE_SHARPE', 'Not Found'),
            'SELF_CORRELATION': check_results.get('SELF_CORRELATION', 'Not Found'),
            'MATCHES_COMPETITION': check_results.get('MATCHES_COMPETITION', 'Not Found') 
        }
        alpha_list.append(data)
        
    return pd.DataFrame(alpha_list) if pandas else alpha_list