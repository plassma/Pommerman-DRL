import numpy as np
import torch
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2
from gym.spaces import Box, Discrete
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models import create_policy

writer = SummaryWriter('results')
N_game = 10
NUM_ACTIONS = 6
RENDER = False
ENV_ID = 'GraphicOVOCompact-v0'

actor_critic = create_policy(
        Box(np.zeros(13440), np.ones(13440)),
        Discrete(6),
        name='pomm',
        nn_kwargs={
            'batch_norm': True,
            'recurrent': False,
            'hidden_size': 512,
            'cnn_config': 'conv5',
        },
        train=True)

actor_critic.load_state_dict(torch.load("trained_models/a2c/GraphicOVOCompact-v0.pt")[0])
actor_critic = actor_critic.cuda()


def evaluate_model(model, opponent_actor=None):
    print("evaluating...")
    model.eval()

    # Make the "Free-For-All" environment using the agent list
    env = PommerEnvWrapperFrameSkip2(num_stack=5, start_pos=0, board=ENV_ID, opponent_actor=opponent_actor)
    # Run the episodes just like OpenAI Gym
    win_cnt = 0
    draw_cnt = 0
    lost_cnt = 0
    actions = np.zeros(6)
    for i_episode in range(N_game):
        obs, opponent_obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            net_out = model(torch.tensor(obs).float().cuda())
            action = net_out.argmax(1).item()
            actions[action] += 1
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
    return obs


sample_input = evaluate_model(actor_critic)
input = torch.tensor(sample_input)
writer.add_graph(actor_critic.cpu(), input.float())
torch.onnx.export(actor_critic, input.float(), f="v2.onnx", export_params=True, opset_version=12, do_constant_folding=True)
