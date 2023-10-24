import numpy as np
import scipy.signal
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import time
import os.path as osp

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
from logx import EpochLogger, setup_logger_kwargs

import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    

class BipedalWalkerHardcoreWrapper(object):
    def __init__(self, action_repeat=3):
        self._env = gym.make("BipedalWalker-v3")
        self.action_repeat = action_repeat
        self.act_noise = 0.3
        self.reward_scale = 5.0
        self.observation_space = self._env.observation_space
        

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        action += self.act_noise * (-2 * np.random.random(4) + 1)
        r = 0.0
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_, x = self._env.step(action)
            if done_ and reward_ == -100:
                reward_ = 0
            r = r + reward_
            if done_ and self.action_repeat != 1:
                return obs_, 0.0, done_, info_, x
            if self.action_repeat == 1:
                return obs_, r, done_, info_, x
        return obs_, self.reward_scale * r, done_, info_, x


def get_last_file_number(directory):
    files = os.listdir(directory)
    max_number = -1
    for file in files:
        if file.startswith("info_") and file.endswith(".csv"):
            try:
                number = int(file.split("_")[1].split(".")[0])
                if number > max_number:
                    max_number = number
            except ValueError:
                continue
    return max_number

def test_no_log_file(env, model, path: str, render_mode="human", max_eps_length=1000, deterministic=False):
    h=False
    if render_mode == "human":
        h=True
    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    last_x, last_y = 0, 0
    
    while not(d) and  ep_len < max_eps_length: 
        # Take deterministic actions at test time 
        a = model.get_action(torch.as_tensor(o, dtype=torch.float32), deterministic=deterministic)
        # print(a)
        o, r, d, _ , info = env.step(a)
        print(o[0])
        if o[0] > 0.9:
            time.sleep(10)
            # input("Enter your value: ")

        x_pos = info.get('x_position', 0)
        y_pos = info.get('y_position', 0)
        
        # print(f"x_pos - last_x: {(x_pos - last_x):.4f}, y_pos - last_y: {(y_pos - last_y):.4f}")

        x_pos = last_x
        y_pos = last_y

        ep_ret += r
        ep_len += 1
    print(ep_len)
    print(env.energy) 



def test_model(env, model, path: str, render_mode="human", max_eps_length=1000, deterministic=False):
    h=False
    if render_mode == "human":
        h=True
    
    from pathlib import Path
    path = Path(path)
    path_to_info_csv = os.path.join(path.parent.parent.absolute(), 'infos')
    path_to_state_csv = os.path.join(path.parent.parent.absolute(), 'states')
    path_to_action_csv = os.path.join(path.parent.parent.absolute(), 'actions')
    path_to_gif = os.path.join(path.parent.parent.absolute(), 'videos')
    path_to_map = os.path.join(path.parent.parent.absolute(), 'maps')
    
    for dir in [path_to_info_csv, path_to_state_csv, path_to_action_csv, path_to_gif, path_to_map]:
        if not os.path.exists(dir):
            os.mkdir(dir)
        
    m = get_last_file_number(path_to_info_csv)
    path_to_info_csv = os.path.join(path_to_info_csv, f'info_{m+1}.csv')
    path_to_state_csv = os.path.join(path_to_state_csv, f'states_{m+1}.csv')
    path_to_action_csv = os.path.join(path_to_action_csv, f'actions_{m+1}.csv')
    path_to_gif = os.path.join(path_to_gif, f'video_{m+1}.gif')
    path_to_map = os.path.join(path_to_map, f'map_{m+1}.png')
    

    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    df_index=0
    xys = []

    while not(d) and  ep_len < max_eps_length: 
        # Take deterministic actions at test time 
        
        a = model.get_action(torch.as_tensor(o, dtype=torch.float32), deterministic=deterministic)
        o, r, d, _ , info = env.step(a)
        print(o[0])
        if o[0] > 0.8:
            time.sleep(5)
        xys.append((info["x_position"], info["y_position"]))
        if df_index == 0:
            df_info = pd.DataFrame(columns=list(info.keys()))
            df_state = pd.DataFrame(columns=list(range(o.shape[0])))
            df_action = pd.DataFrame(columns=list(range(a.shape[0])))
        
        df_info.loc[df_index] = info
        df_state.loc[df_index] = {e:v for e,v in zip(range(o.shape[0]),o)}
        df_action.loc[df_index] = {e:v for e,v in zip(range(a.shape[0]),a)}

        ep_ret += r
        ep_len += 1 
        df_index+=1
    print(f"ep_len: {ep_len}")
    df_info.to_csv(path_to_info_csv, index=False)
    df_state.to_csv(path_to_state_csv, index=False)
    df_action.to_csv(path_to_action_csv, index=False)

    
    def symmetrize_axes(axes):
        y_max = np.abs(axes.get_ylim()).max() + 1
        axes.set_ylim(ymin=-y_max, ymax=y_max)
        
        x_max = np.abs(axes.get_xlim()).max() + 1
        axes.set_xlim(xmin=-x_max, xmax=x_max)
   
    plt.figure()
    for i, position in enumerate(xys[:-1]):
        x, y = position
        x_, y_ = xys[i+1]
        plt.plot((x, x_),(y, y_), "b")
    
    ax = plt.gca()
    symmetrize_axes(ax)
    plt.savefig(path_to_map)
    plt.close()



def test_two_cams(env, model, path: str, render_mode="human", max_eps_length=1000, deterministic=False):
    h=False
    if render_mode == "human":
        h=True
    
    from pathlib import Path
    path = Path(path)
    path_to_info_csv = os.path.join(path.parent.parent.absolute(), 'infos')
    path_to_state_csv = os.path.join(path.parent.parent.absolute(), 'states')
    path_to_action_csv = os.path.join(path.parent.parent.absolute(), 'actions')
    path_to_gif = os.path.join(path.parent.parent.absolute(), 'videos')
    path_to_map = os.path.join(path.parent.parent.absolute(), 'maps')
    
    for dir in [path_to_info_csv, path_to_state_csv, path_to_action_csv, path_to_gif, path_to_map]:
        if not os.path.exists(dir):
            os.mkdir(dir)
        
    m = get_last_file_number(path_to_info_csv)
    path_to_info_csv = os.path.join(path_to_info_csv, f'info_{m+1}.csv')
    path_to_state_csv = os.path.join(path_to_state_csv, f'states_{m+1}.csv')
    path_to_action_csv = os.path.join(path_to_action_csv, f'actions_{m+1}.csv')
    path_to_gif_0 = os.path.join(path_to_gif, f'video_{m+1}_cam_0.gif')
    path_to_gif_1 = os.path.join(path_to_gif, f'video_{m+1}_cam_1.gif')
    path_to_map = os.path.join(path_to_map, f'map_{m+1}.png')
    

    o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
    df_index=0
    xys = []
    with imageio.get_writer(path_to_gif_0, mode='I', duration=22) as writer_0:
        with imageio.get_writer(path_to_gif_1, mode='I', duration=22) as writer_1:
            frames = []
            while not(d) and  ep_len < max_eps_length: 
                # Take deterministic actions at test time 
                
                a = model.get_action(torch.as_tensor(o, dtype=torch.float32), deterministic=deterministic)
                o, r, d, _ , info = env.step(a)
                xys.append((info["x_position"], info["y_position"]))
                if df_index == 0:
                    df_info = pd.DataFrame(columns=list(info.keys()))
                    df_state = pd.DataFrame(columns=list(range(o.shape[0])))
                    df_action = pd.DataFrame(columns=list(range(a.shape[0])))
                
                df_info.loc[df_index] = info
                df_state.loc[df_index] = {e:v for e,v in zip(range(o.shape[0]),o)}
                df_action.loc[df_index] = {e:v for e,v in zip(range(a.shape[0]),a)}
                # frame = env.render()
                env.get_viewer("rgb_array").render(camera_id=0)
                data = env.get_viewer(env.render_mode).read_pixels(depth=False)
                frame0 = data[::-1, :, :]

                env.get_viewer("rgb_array").render(camera_id=1)
                data = env.get_viewer(env.render_mode).read_pixels(depth=False)
                frame1 = data[::-1, :, :]
                
                writer_0.append_data(frame0)
                writer_1.append_data(frame1)
                
                ep_ret += r
                ep_len += 1 
                df_index+=1
    print(f"ep_len: {ep_len}")
    df_info.to_csv(path_to_info_csv, index=False)
    df_state.to_csv(path_to_state_csv, index=False)
    df_action.to_csv(path_to_action_csv, index=False)
    # if ep_len > 4500:
    #     if not h:
    #         imageio.mimwrite(path_to_gif, frames, duration=22)
    
    def symmetrize_axes(axes):
        y_max = np.abs(axes.get_ylim()).max() + 1
        axes.set_ylim(ymin=-y_max, ymax=y_max)
        
        x_max = np.abs(axes.get_xlim()).max() + 1
        axes.set_xlim(xmin=-x_max, xmax=x_max)
   
    plt.figure()
    for i, position in enumerate(xys[:-1]):
        x, y = position
        x_, y_ = xys[i+1]
        plt.plot((x, x_),(y, y_), "b")
    
    ax = plt.gca()
    symmetrize_axes(ax)
    plt.savefig(path_to_map)
    plt.close()

 

# def test_model_human(env, path: str, mode="human"):
#     env = gym.make(env, healthy_z_range=(0.3, 1.0), render_mode=mode)
#     ac = torch.load(path).to("cpu")
    
#     from pathlib import Path
#     path = Path(path)
#     path_to_info_csv = os.path.join(path.parent.parent.absolute(), 'infos')
#     path_to_state_csv = os.path.join(path.parent.parent.absolute(), 'states')
#     path_to_action_csv = os.path.join(path.parent.parent.absolute(), 'actions')
#     path_to_gif = os.path.join(path.parent.parent.absolute(), 'videos')
#     path_to_map = os.path.join(path.parent.parent.absolute(), 'maps')
    
#     for dir in [path_to_info_csv, path_to_state_csv, path_to_action_csv, path_to_gif, path_to_map]:
#         if not os.path.exists(dir):
#             os.mkdir(dir)
        
#     m = get_last_file_number(path_to_info_csv)
#     path_to_info_csv = os.path.join(path_to_info_csv, f'info_{m+1}.csv')
#     path_to_state_csv = os.path.join(path_to_state_csv, f'states_{m+1}.csv')
#     path_to_action_csv = os.path.join(path_to_action_csv, f'actions_{m+1}.csv')
#     path_to_gif = os.path.join(path_to_gif, f'video_{m+1}.gif')
#     path_to_map = os.path.join(path_to_map, f'map_{m+1}.png')
    

#     o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
#     df_index=0
#     xys = []
#     while not(d): # and  ep_len < 1000
#         # Take deterministic actions at test time 
#         a = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=False)
#         o, r, d, _ , info = env.step(a)

#         xys.append((info["x_position"], info["y_position"]))
#         if df_index == 0:
#             df_info = pd.DataFrame(columns=list(info.keys()))
#             df_state = pd.DataFrame(columns=list(range(o.shape[0])))
#             df_action = pd.DataFrame(columns=list(range(a.shape[0])))
        
#         df_info.loc[df_index] = info
#         df_state.loc[df_index] = {e:v for e,v in zip(range(o.shape[0]),o)}
#         df_action.loc[df_index] = {e:v for e,v in zip(range(a.shape[0]),a)}
#         frame = env.render()
        
#         ep_ret += r
#         ep_len += 1 
#         df_index+=1

#     df_info.to_csv(path_to_info_csv, index=False)
#     df_state.to_csv(path_to_state_csv, index=False)
#     df_action.to_csv(path_to_action_csv, index=False)
    
#     def symmetrize_axes(axes):
#         y_max = np.abs(axes.get_ylim()).max() + 1
#         axes.set_ylim(ymin=-y_max, ymax=y_max)
        
#         x_max = np.abs(axes.get_xlim()).max() + 1
#         axes.set_xlim(xmin=-x_max, xmax=x_max)
    
#     for i, position in enumerate(xys[:-1]):
#         x, y = position
#         x_, y_ = xys[i+1]
#         plt.plot((x, x_),(y, y_), "b")
    
#     ax = plt.gca()
#     symmetrize_axes(ax)
#     plt.savefig(path_to_map)

# def test_model(env, path: str, mode="rgb_array"):
#     env = gym.make(env, healthy_z_range=(0.3, 1.0), render_mode=mode)
#     ac = torch.load(path).to("cpu")
    
#     from pathlib import Path
#     path = Path(path)
#     path_to_info_csv = os.path.join(path.parent.parent.absolute(), 'infos')
#     path_to_state_csv = os.path.join(path.parent.parent.absolute(), 'states')
#     path_to_action_csv = os.path.join(path.parent.parent.absolute(), 'actions')
#     path_to_gif = os.path.join(path.parent.parent.absolute(), 'videos')
#     path_to_map = os.path.join(path.parent.parent.absolute(), 'maps')
    
#     for dir in [path_to_info_csv, path_to_state_csv, path_to_action_csv, path_to_gif, path_to_map]:
#         if not os.path.exists(dir):
#             os.mkdir(dir)
        
#     m = get_last_file_number(path_to_info_csv)
#     path_to_info_csv = os.path.join(path_to_info_csv, f'info_{m+1}.csv')
#     path_to_state_csv = os.path.join(path_to_state_csv, f'states_{m+1}.csv')
#     path_to_action_csv = os.path.join(path_to_action_csv, f'actions_{m+1}.csv')
#     path_to_gif = os.path.join(path_to_gif, f'video_{m+1}.gif')
#     path_to_map = os.path.join(path_to_map, f'map_{m+1}.png')
    

#     o, d, ep_ret, ep_len = env.reset()[0], False, 0, 0
#     df_index=0
#     frames = []
#     xys = []
#     while not(d) and  ep_len < 1000:
#         # Take deterministic actions at test time 
#         a = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic=False)
#         o, r, d, _ , info = env.step(a)
#         xys.append((info["x_position"], info["y_position"]))
#         if df_index == 0:
#             df_info = pd.DataFrame(columns=list(info.keys()))
#             df_state = pd.DataFrame(columns=list(range(o.shape[0])))
#             df_action = pd.DataFrame(columns=list(range(a.shape[0])))
        
#         df_info.loc[df_index] = info
#         df_state.loc[df_index] = {e:v for e,v in zip(range(o.shape[0]),o)}
#         df_action.loc[df_index] = {e:v for e,v in zip(range(a.shape[0]),a)}
#         frame = env.render()
#         frames.append(frame)
        
#         ep_ret += r
#         ep_len += 1 
#         df_index+=1

#     df_info.to_csv(path_to_info_csv, index=False)
#     df_state.to_csv(path_to_state_csv, index=False)
#     df_action.to_csv(path_to_action_csv, index=False)
#     imageio.mimwrite(path_to_gif, frames, duration=22)
    
#     def symmetrize_axes(axes):
#         y_max = np.abs(axes.get_ylim()).max() + 1
#         axes.set_ylim(ymin=-y_max, ymax=y_max)
        
#         x_max = np.abs(axes.get_xlim()).max() + 1
#         axes.set_xlim(xmin=-x_max, xmax=x_max)
    
#     plt.figure()
#     for i, position in enumerate(xys[:-1]):
#         x, y = position
#         x_, y_ = xys[i+1]
#         plt.plot((x, x_),(y, y_), "b")
    
#     ax = plt.gca()
#     symmetrize_axes(ax)
#     plt.savefig(path_to_map)
