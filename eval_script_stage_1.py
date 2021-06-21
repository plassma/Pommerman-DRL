import numpy as np
import torch
import onnx
from onnx2pytorch import ConvertModel
import argparse
import sys
from gym import logger as gymlogger

# Environment import and set logger level to display error only
gymlogger.set_level(40)  # error only
# ignore prints to stdout of imports
save_stdout = sys.stdout
sys.stdout = open('trash', 'w')
import os

os.system("git clone https://github.com/MultiAgentLearning/playground ./pommer_setup")
os.system("pip install -U ./pommer_setup")
os.system('rm -rf ./pommer_setup')
os.system("git clone https://github.com/RLCommunity/graphic_pomme_env ./graphic_pomme_env")
os.system("pip install -U ./graphic_pomme_env")
os.system('rm -rf ./graphic_pomme_env')
sys.stdout = save_stdout
from graphic_pomme_env import graphic_pomme_env
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2

# Seed random number generators
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None

if __name__ == "__main__":
    N_EPISODES = 10
    actions = np.zeros(6)

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Network
    net = ConvertModel(onnx.load(model_file), experimental=True)
    net = net.to(device)
    net.eval()

    win_count = 0.0
    env = PommerEnvWrapperFrameSkip2(num_stack=5, start_pos=0, board='GraphicOVOCompact-v0')

    for i in range(N_EPISODES):
        if seed is not None:
            seed = np.random.randint(1e7)

        done = False
        obs, opponent_obs = env.reset()
        while not done:
            obs = torch.from_numpy(np.array(obs)).float().to(device)
            net_out = net(obs).detach().cpu().numpy()
            action = np.argmax(net_out)
            actions[action] += 1
            agent_step, opponent_step = env.step(action)
            obs, r, done, info = agent_step

        if r > 0:
            win_count += 1

    print(win_count / N_EPISODES)
