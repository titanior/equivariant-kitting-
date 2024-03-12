import numpy as np
import argparse
import os
import sys
sys.path.append('./')
sys.path.append('..')
sys.path.append('PATH_TO_Help_Hans_rl_envs')
from raven.dataset import Dataset
from raven.gripper_env import Environment
from raven.gripper_block_insertion import BlockInsertion
import matplotlib.pyplot as plt
from raven import utils_
from raven import cameras
enc_cls = Environment
env = enc_cls('./raven/assets/',
                disp=True,
                shared_memory=False,
                hz=480)

task = BlockInsertion(continuous=False)
env.set_task(task)
obs = env.reset()
while 1:
    act = dict()
    act["pose0"] = (np.array([0.546875, 0.33125 , 0.07]),np.array([0., 0., 0.64278761, 0.76604444]))
    act["pose1"] = (np.array([0.33741313, 0.14475863, 0.07]),np.array([-0.       ,  0.       , -0.5      , -0.8660254]))
    obs, reward, done, info = env.step(act)
    cmap, hmap = utils_.get_fused_heightmap(
            obs, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    # print(hmap.shape)
    # hmap1 = hmap[0:160,:]
    # print(hmap1.shape)
    plt.imshow(hmap, cmap='gray')
    plt.show()
    obs = env.reset()