from typing import List
import random

def single_host(
        core_state:     List[float], 
        arrive_time:    float, 
        task_size:      List[float]
    )->List[int]:

    return [random.randint(0,len(core_state)-1) for _ in task_size]

def multi_hosts(
        cores_state:    List[List[float]], 
        arrive_time:    float, 
        task_locations: List[List[int]], 
        task_size:      List[float],
        cost:           float
    )->int:
    
    return random.randint(0,len(cores_state)-1)
