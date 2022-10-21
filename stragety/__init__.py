from typing import List

def single_host(
        core_state:     List[float], 
        arrive_time:    float, 
        task_size:      List[float]
    )->List[int]:

    M=(max(task_size)+min(task_size)*1.2)*max(1,len(task_size)/len(core_state))

    task_order=sorted(
        [[task_size[i],i] for i in range(len(task_size))],
        key=lambda i:task_size[i[1]],
        reverse=True
    )
    start=0
    while start<len(task_order)-1:
        if task_order[start][0]+task_size[task_order[-1][1]]<M:
            task_order[start].append(task_order.pop()[1])
            task_order[start][0]+=task_size[task_order[start][-1]]
        else:start+=1

    core_order=sorted(
        list(range(len(core_state))),
        key=lambda i:max(core_state[i],arrive_time)
    )
    ret=[None]*len(task_size)
    start=0
    for i in task_order:
        for j in range(1,len(i)):
            ret[i[j]]=core_order[start]
        start+=1
    return ret


def multi_hosts(
        cores_state:    List[List[float]], 
        arrive_time:    float, 
        task_locations: List[List[int]], 
        task_size:      List[float],
        cost:           float
    )->int:

    arg1=0.80
    
    hosts_size=[0.]*len(cores_state)
    for locs,size in zip(task_locations,task_size):
        for loc in locs:
            hosts_size[loc]+=size
    order=sorted(
        list(range(len(cores_state))),
        key=lambda i:hosts_size[i],
        reverse=True
    )

    for i in range(1,len(order)):
        if hosts_size[order[i]]<hosts_size[order[0]]*arg1:
            break
    order=order[:i]

    min_time,ret=float("inf"),None
    for h in order:
        tasksize=[
            task_size[i]
            if h in task_locations[i] 
            else cost*task_size[i] 
            for i in range(len(task_size))
        ]
        tmp_assignment=single_host(cores_state[h],arrive_time,task_size)
        
        tmp_start_line=arrive_time
        for target in tmp_assignment:
            tmp_start_line=max(
                tmp_start_line,
                cores_state[h][target]
            )

        for target,size in zip(tmp_assignment,tasksize):
            cores_state[h][target]=max(cores_state[h][target],tmp_start_line)
            cores_state[h][target]+=size
        
        max_finish_time=max(cores_state[h])
        if max_finish_time<min_time:
            min_time=max_finish_time
            ret=h
    
    return ret