from typing import List,Callable
import numpy as np, random, copy

class Frame:

    class task:
        def __init__(self,size:float,arrive_time:float,index:int,father:'Frame.Job'):
            self.size:float=size
            self.arrive_time:float=arrive_time
            self.father:Frame.Job=father
            self.index:int=index
        
        def __eq__(self, other:'Frame.task')->bool:
            return self.father==other.father and self.index==other.index
        
        def __hash__(self)->int:
            return hash((self.father,self.index))
    
    
    class Job:
        def __init__(self,arrive_time:float,index:int,num_of_hosts:int):
            self.arrive_time:float=arrive_time
            self.tasks:List[Frame.task]=[
                Frame.task(random.randint(20,100),self.arrive_time,_,self)
                for _ in range(random.randint(5,15))
            ]
            self.task_locations:List[List[int]]=[
                random.sample(range(num_of_hosts),random.randint(3,5))
                for _ in self.tasks
            ]
            self.index:int=index
        
        def __eq__(self,other:'Frame.Job')->bool:
            return self.index==other.index
        
        def __hash__(self)->int:
            return self.index



    def __init__(self, 
                 single_host:Callable[[List[float],float,List[float]],List[int]], 
                 multi_hosts:Callable[[List[List[float]],float,List[List[int]],List[float],float],int],
                 lmd:float=0.5,
                 num_of_hosts:int=20,
                 num_of_cores:int=25,
                 cost:float=6/5):
        self.t:float=0.
        self.lmd:float=lmd

        self.num_of_hosts:int=num_of_hosts
        self.num_of_cores:int=num_of_cores
        
        self.cores_on_hosts=[
            [0.]*self.num_of_cores 
            for _ in range(self.num_of_hosts)
        ]
        self.job_count:int=0

        self.cost:float=cost

        self.single_host:Callable[
            [
                List[float],
                float, 
                List[float]
            ],
            List[int]
        ]=single_host
        self.multi_hosts:Callable[
            [
                List[List[float]],
                float,
                List[List[int]],
                List[float],
                float
            ],
            int
        ]=multi_hosts

        self.step_j:Frame.Job
        self.step_host:int
        self.step_assignment:List[int]
        self.step_start_line:float

    def step(self)->None:
        self.t+=np.random.exponential(1/self.lmd)
        self.step_j=Frame.Job(self.t,self.job_count,self.num_of_hosts)
        self.job_count+=1

        self.step_host=self.multi_hosts(
            copy.deepcopy(self.cores_on_hosts),
            self.step_j.arrive_time,
            copy.deepcopy(self.step_j.task_locations),
            [_.size for _ in self.step_j.tasks],
            self.cost
        )

        tasksize=[
            self.step_j.tasks[task_index].size 
            if self.step_host in self.step_j.task_locations[task_index] 
            else self.cost*(self.step_j.tasks[task_index].size) 
            for task_index in range(len(self.step_j.tasks))
        ]

        self.step_assignment=self.single_host(
            copy.deepcopy(self.cores_on_hosts[self.step_host]),
            self.step_j.arrive_time,
            tasksize
        )

        self.step_start_line=self.step_j.arrive_time
        for target in self.step_assignment:
            self.step_start_line=max(
                self.step_start_line,
                self.cores_on_hosts[self.step_host][target]
            )

        for target,size in zip(self.step_assignment,tasksize):
            self.cores_on_hosts[self.step_host][target]=max(
                self.cores_on_hosts[self.step_host][target],
                self.step_start_line
            )
            self.cores_on_hosts[self.step_host][target]+=size


