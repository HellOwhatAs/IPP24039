from typing import List
import random, numpy as np
from sko.GA import GA

def single_host(core_state:List[float], 
                arrive_time:float, 
                task_size:List[float])->List[int]:
    _cs=np.array(core_state)
    def obj(_x:np.ndarray):
        x=_x.astype(np.int32)
        start_line=arrive_time
        cs=_cs.copy()
        for i in x:
            start_line=max(cs[i],start_line)
        for ind in range(x.shape[0]):
            cs[x[ind]]=max(cs[x[ind]],start_line)
            cs[x[ind]]+=task_size[ind]
        ret=start_line
        for i in x:
            ret=max(ret,cs[i])
        return ret
    ga=GA(
        func=obj,
        n_dim=len(task_size),
        lb=0,
        ub=len(core_state)-1,
        precision=1
    )

    return [round(i) for i in ga.run()[0]]
    # return [random.randint(0,len(core_state)-1) for _ in task_size]

def multi_hosts(cores_state:List[List[float]], 
                arrive_time:float, 
                task_locations:List[List[int]], 
                task_size:List[float],
                cost:float)->int:
    return np.argmin(np.mean(np.maximum(cores_state,arrive_time),axis=1))




# demo_func = lambda x: -(x[0]+x[1]+x[2])
# ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=0, ub=10, precision=1)
# best_x, best_y = ga.run()
# print('best_x:', best_x, '\n', 'best_y:', best_y)