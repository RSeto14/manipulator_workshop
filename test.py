import gym
from gym import wrappers
from math import cos, sin, acos, asin, atan, pi
import numpy as np
from controller import Controller
env = gym.make('Kinova-v0')
obs = env.reset()
action = [0,0,0]
target = [0.65,0.8]
controller = Controller(dt=0.005)
start = 1000
end = 1000

for step in range(start + 10000 + end):
    
    if step < start:
        env.render()
        target = [0.65,0.8]
        theta = obs
        action = controller.control(theta=np.array(theta), x_target=np.array(target), kp=3, kd=0.1)
        obs, reward, done, info = env.step(action)

    elif start <= step < start +10000:
        env.render()
        t = step - start

        #tを媒介変数にしたx,z###############################################

        target = [0.1*sin(2*pi*t/1000) + 0.65, 0.1*cos(2*pi*t/1000) + 0.7]
        
        ##################################################################
        theta = obs
        action = controller.control(theta=np.array(theta), x_target=np.array(target), kp=3, kd=0.1)
        obs, reward, done, info = env.step(action)
    else:
        env.render()

        theta = obs
        action = controller.control(theta=np.array(theta), x_target=np.array(target), kp=3, kd=0.1)
        obs, reward, done, info = env.step(action)



else:
    print("finished")