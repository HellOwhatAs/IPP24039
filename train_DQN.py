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

dqn=DQN(20,20*25,0)
dqnm=DQN_Manager(dqn)

while True:
    random.seed(seed),np.random.seed(seed)
    frame=Frame(
        single_host,
        dqnm.multi_hosts,
        lmd=lmd
    )

    y1,y2,rs=[],[],[]
    for i in tqdm(range(jobnum),title):
        frame.step()
        finish_time=max(
            frame.cores_on_hosts[frame.step_host][c] 
            for c in frame.step_assignment
        )
        r = frame.step_j.arrive_time-finish_time
        dqnm.learn(r)
        y1.append(frame.step_start_line-frame.step_j.arrive_time)
        y2.append(finish_time-frame.step_start_line)
        rs.append(r)
    plt.figure()
    plt.title(title)
    plt.plot(y1,label="wait time")
    plt.plot(y2,label="execute time")
    plt.plot(rs,label="reward")
    plt.legend()

    
    plt.pause(1)
    plt.close()
    # plt.show()