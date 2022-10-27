from typing import List
import numpy as np

def single_host(
        core_state:     List[float], 
        arrive_time:    float, 
        task_size:      List[float]
    )->List[int]:

    return [np.argmin(np.maximum(core_state,arrive_time))]*len(task_size)

def multi_hosts(
        cores_state:    List[List[float]], 
        arrive_time:    float, 
        task_locations: List[List[int]], 
        task_size:      List[float],
        cost:           float
    )->int:
    
    return np.argmin(np.mean(np.maximum(cores_state,arrive_time),axis=1))
