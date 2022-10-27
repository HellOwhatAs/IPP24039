from typing import List
import random, numpy as np

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
    
    return np.argmin(np.mean(np.maximum(cores_state,arrive_time),axis=1))
