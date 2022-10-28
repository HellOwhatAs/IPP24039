import random, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frame import Frame

seed=0
lmd=0.6
jobnum=2000

# dqn
from stragety.DQN import single_host, DQN, DQN_Manager

title="DQN stragety"
random.seed(seed),np.random.seed(seed)

dqn=DQN()
dqnm=DQN_Manager(dqn)
while True:
    frame=Frame(
        single_host,
        dqnm.multi_hosts,
        lmd=lmd
    )

    y1,y2=[],[]
    for i in tqdm(range(jobnum),title):
        frame.step()
        finish_time=max(
            frame.cores_on_hosts[frame.step_host][c] 
            for c in frame.step_assignment
        )
        r = sum(t.size for t in frame.step_j.tasks)/(finish_time-frame.step_j.arrive_time)
        dqnm.learn(r)
        y1.append(frame.step_start_line-frame.step_j.arrive_time)
        y2.append(finish_time-frame.step_start_line)
    plt.figure()
    plt.title(title)
    plt.plot(y1,label="wait time")
    plt.plot(y2,label="execute time")
    plt.legend()

    
    plt.pause(1)
    plt.close()