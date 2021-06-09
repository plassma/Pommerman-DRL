from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2
from pommerman.agents import BaseAgent

from envs import make_vec_envs
from mapper_net import MappingNet
from gym.spaces import Box, Discrete
from models import create_policy
import torch
from torch import nn
import numpy as np
from envs.pommerman import featurize, DEFAULT_FEATURE_CONFIG
from pommerman.agents import SimpleAgent


N_game = 50
NUM_ACTIONS = 6
RENDER = False
ENV_ID = 'GraphicOVOCompact-v0'

actor_critic = create_policy(
        Box(np.zeros(327), np.ones(327)),
        Discrete(6),
        name='pomm',
        nn_kwargs={
            'batch_norm': True,
            'recurrent': False,
            'hidden_size': 512,
        },
        train=True)

actor_critic.load_state_dict(torch.load("trained_models/a2c/GraphicOVOCompact-v0.pt")[0])

mapper = MappingNet(5, 327)
mapper.load_state_dict(torch.load("framestack_to_raw_mapper_9.pth"))

class Destroyer(nn.Module):
    def __init__(self, mapper, a2c):
        super().__init__()
        self.mapper = mapper
        self.a2c = a2c

    def forward(self, x):
        raw_state = self.mapper(x)
        return self.a2c.act(raw_state, torch.zeros(1).cuda(), torch.zeros(1).cuda(), deterministic=True)


def evaluate_model(model, opponent_actor=None):
    print("evaluating...")

    # Make the "Free-For-All" environment using the agent list
    env = PommerEnvWrapperFrameSkip2(num_stack=5, start_pos=0, board=ENV_ID, opponent_actor=opponent_actor)
    # Run the episodes just like OpenAI Gym
    win_cnt = 0
    draw_cnt = 0
    lost_cnt = 0
    for i_episode in range(N_game):
        obs, opponent_obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            if RENDER: env.env.render()

            if isinstance(model, BaseAgent):
                obs = env.get_last_step_raw()
                action = model.act(obs[0], NUM_ACTIONS)
            else:
                #obs = env.get_last_step_raw()
                #obs = torch.tensor(featurize(obs[0], 42, DEFAULT_FEATURE_CONFIG)).float().unsqueeze(0).cuda()
                #net_out = actor_critic.act(obs, torch.zeros(1).cuda(), torch.ones(1).cuda(), deterministic=True)
                net_out = model(obs)  #.detach().cpu().numpy()
                #action = np.argmax(net_out)
                action = net_out[1].item()
            agent_step, opponent_step = env.step(action)
            obs, r, done, info = agent_step
            step_cnt += 1

        if r > 0:
            win_cnt += 1
        elif step_cnt >= 800:
            draw_cnt += 1
        else:
            lost_cnt = lost_cnt + 1
        # print('Episode {} finished'.format(i_episode))
    print('win:', win_cnt, 'draw_cnt:', draw_cnt, 'lose_cnt:', lost_cnt)
    print('\n')
    env.env.close()

def evaluate_model_(model, opponent_actor=None):
    print("evaluating...")

    # Make the "Free-For-All" environment using the agent list
    env = make_vec_envs(
            ENV_ID, 0, 2, 0.99,
            True, 1, "/tmp/gym/_eval", False, "cuda",
            allow_early_resets=True, eval=True)
    # Run the episodes just like OpenAI Gym
    win_cnt = 0
    draw_cnt = 0
    lost_cnt = 0
    for i_episode in range(N_game):
        obs = env.reset()
        done = torch.tensor([False])
        step_cnt = 0
        while not done.all():
            if RENDER: env.env.render()

            if isinstance(model, BaseAgent):
                obs = env.get_last_step_raw()
                action = model.act(obs[0], NUM_ACTIONS)
            else:
                #obs = torch.tensor(featurize(obs[0], 0, DEFAULT_FEATURE_CONFIG)).float().unsqueeze(0).cuda()
                net_out = actor_critic.act(obs, torch.zeros(1).cuda(), torch.zeros(1).cuda(), deterministic=True)
                #net_out = model(obs)  #.detach().cpu().numpy()
                #action = np.argmax(net_out)
                action = net_out[1]
            agent_step = env.step(action)
            obs, r, done, info = agent_step
            step_cnt += 1
        win_cnt += (r > 0).sum()
        if r > 0:
            win_cnt += 1
        elif step_cnt >= 800:
            draw_cnt += 1
        else:
            lost_cnt = lost_cnt + 1
        # print('Episode {} finished'.format(i_episode))
    print('win:', win_cnt, 'draw_cnt:', draw_cnt, 'lose_cnt:', lost_cnt)
    print('\n')

evaluate_model(Destroyer(mapper, actor_critic).cuda())
