import copy
from collections import deque
import Maze_Generation.gen as maze_gen
from q_maze import ACTIONS, Qmaze
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchinfo import summary

import Maze_Generation.gen as maze_gen
# exploration factor

class QModel(nn.Module):
    def __init__(self, maze_size):
        super().__init__()
        self.fc1 = nn.Linear(maze_size, maze_size)
        self.fc2 = nn.Linear(maze_size, len(ACTIONS))
        self.prelu = nn.PReLU()
    def forward(self, x):
        #x = torch.flatten(x)
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.prelu(self.fc1(x))
        x = self.fc2(x)
        return x


def follows_path(qmaze, model, optim_path):
    qmaze.reset((0,0))
    model.eval()
    for move in optim_path:
        if tuple(qmaze.state[0:2]) != move:
            return False
        envstate = torch.from_numpy(qmaze.observe()).float()
        envstate = envstate.cuda()
        q = model(envstate)
        q = q.data.to('cpu').numpy()
        action = np.argmax(q)
        qmaze.act(action)
    return True

def train_model(qmaze: Qmaze, optim_path):
    n_epoch = 2000
    lr = 1e-3
    epsilon = 0.1
    mem_size = 1000
    batch_size = 200
    gamma = 0.9
    symc_freq = 500
    total_count = 0
    h_size = qmaze.maze.size//2
    replay = deque(maxlen = mem_size)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = QModel(qmaze.maze.size)
    if device == 'cuda':
        model = model.cuda()
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    win_history = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    losses = []
    win_rate = 0.0
    for i in range(n_epoch):
        qmaze.reset((0,0))
        game_over = False
        optimizer.zero_grad()
        running_losses = []
        envstate = torch.from_numpy(qmaze.observe()).float()
        if device == "cuda":
            envstate = envstate.cuda()
        n_episodes = 0

        while not game_over:
            total_count += 1
            valid_actions = qmaze.valid_actions()
            if not valid_actions:
                print('no valid')
                break
            qval = model(envstate)
            qval_ = qval.data.to('cpu').numpy()
            if random.random() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                action = np.argmax(qval_)

            # apply_action, get reward and new envstate
            new_envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win' or game_status == 'lose':
                win_history.append(game_status=='win')
                game_over= True
            else:
                game_over = False
            new_envstate = torch.from_numpy(new_envstate).float()
            if device == "cuda":
                new_envstate = new_envstate.cuda()

            replay.append([envstate, action, reward, new_envstate, game_over])
            envstate = new_envstate
            n_episodes +=1
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                envstate_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                new_envstate_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                game_over_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                if device == 'cuda':
                    action_batch = action_batch.cuda()
                    reward_batch = reward_batch.cuda()
                    game_over_batch = game_over_batch.cuda()

                Q1 = model(envstate_batch)
                with torch.no_grad():
                    Q2 = model2(new_envstate_batch)

                y = reward_batch + gamma*((1-game_over_batch) * torch.max(Q2, dim=1)[0])
                x = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = criterion(x, y.detach())
                optimizer.zero_grad()
                loss.backward()
                running_losses.append(loss.item())
                optimizer.step()

                if total_count % symc_freq == 0:
                    model2.load_state_dict(model.state_dict())

        print('epoch: ' + str(i+1))
        print('loss: ' + str(np.mean(running_losses)))
        losses.append(np.mean(running_losses))
        if len(win_history) > h_size:
            win_rate = sum(win_history[-h_size:]) / h_size
        print('win rate: ' + str(win_rate))
        if win_rate == 1.0 and follows_path(qmaze, model2, optim_path):
            print("Solved to perfection")
            break

    return

if __name__  == '__main__':
    test = maze_gen.find_shortest_path(maze_gen.maze, (0,0), (9,9))
    train_model(Qmaze(maze_gen.maze), test)
