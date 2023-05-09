import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from math import cos, sin, acos, asin, atan, pi
import numpy as np

class KinovaEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'Gen3Robotiq.xml', 2)

    def step(self, a):
        
        self.do_simulation(a, 5)
        #self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        done = False
        reward = 1
        info = {}
        return obs, reward,done,info

    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    
    def reset_model(self):
        
        target_pos = [0.65,0.8,pi/2]
        ###########################
        h = 0.2848
        La = 0.4208
        Lb = 0.3143
        Lc = 0.1674

        #theta2,theta4,theta6
        target_theta = [0,0,0]

        xh = target_pos[0]
        zh = target_pos[1] - h
        phi = target_pos[2]

        x6  = xh - Lc*sin(phi)
        z_6 = zh - Lc*sin(phi)
        
        target_theta[1] = pi -acos((La**2 + Lb**2 - x6**2 - z_6**2)/(2*La*Lb))
        alpha = atan(z_6/x6)
        beta = acos((La**2 + x6**2 + z_6**2 - Lb**2)/(2*La*((x6**2 + z_6**2)**(1/2))))
        target_theta[0] = pi/2 - alpha - beta
        target_theta[2] = phi - target_theta[0] - target_theta[1]
        ###########################
        qpos = np.array(target_theta)
        qvel = np.array([0,0,0])#初期速度
        self.set_state(qpos,qvel)
        #print(qpos,qvel)
        return self._get_obs()
    """
    def reset_model(self):
        #qpos = np.array([0,0.5,0.5])#初期角度
        target_theta = [pi/6,pi/6,pi/6]
    
        qpos = np.array(target_theta)
        qvel = np.array([0,0,0])#初期速度
        self.set_state(qpos,qvel)

        return self._get_obs()
    """

    def _get_obs(self):
        print("手先位置,角速度[x,y,z]:",self.sim.data.get_site_xpos("hand"),self.sim.data.qvel)
        return np.concatenate([
            #np.cos(theta),
            #np.sin(theta),
            #self.sim.data.qpos.flat[2:],
            #self.sim.data.qvel.flat[:2],
            self.sim.data.qpos,#関節角度
            
        ])
