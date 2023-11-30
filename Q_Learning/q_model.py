import copy
from collections import deque
from Q_Learning.q_maze import ACTIONS, Qmaze, show
import random
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim


# Define the nn
class QModel(nn.Module):
    def __init__(self, maze_size):
        super().__init__()
        self.fc1 = nn.Linear(maze_size, maze_size)
        self.fc2 = nn.Linear(maze_size, len(ACTIONS))
        self.prelu = nn.PReLU()

    def forward(self, x):
        # take in a flattened image and process through the following layers
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.prelu(self.fc1(x))
        x = self.fc2(x)
        return x

# ver
def follows_path(qmaze, model, optim_path, device):
    # reset player back to start
    qmaze.reset((0, 0))
    model.eval()
    for move in optim_path:
        # verifies chosen move matches the optimal path
        if tuple(qmaze.state[0:2]) != move:
            return False
        envstate = torch.from_numpy(qmaze.observe()).float()
        if device == 'cuda':
            envstate = envstate.cuda()
        q = model(envstate)
        q = q.data.to('cpu').numpy()
        action = np.argmax(q)
        qmaze.act(action)
    return True


# play a game out in its entirety
def play_game(qmaze, model, device):
    # reset back to 0,0
    qmaze.reset((0, 0))
    model.eval()
    while 1:
        envstate = torch.from_numpy(qmaze.observe()).float()
        if device == 'cuda':
            envstate = envstate.cuda()
        q = model(envstate)
        q = q.data.to('cpu').numpy()

        # get best predicted next move
        action = np.argmax(q)
        # act on it
        _, _, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def train_model(qmaze: Qmaze, optim_path, **kwargs):
    n_epoch = kwargs['n_epoch']
    lr = kwargs['lr']
    epsilon = kwargs['epsilon']
    epsilon_decay_factor = kwargs['epsilon_decay_factor']
    mem_size = kwargs['mem_size']
    batch_size = kwargs['batch_size']
    gamma = kwargs['gamma']
    sync_freq = kwargs['sync_freq']
    min_epochs = kwargs['min_epochs']
    total_count = 0

    # this type of queue automatically removes oldest items when it gets to maxlen
    replay = deque(maxlen=mem_size)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"Device: {device}")
    # define model 1
    model = QModel(qmaze.maze.size)
    if device == 'cuda':
        model = model.cuda()
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    win_history = []
    losses = []
    win_rate = 0.0
    for i in range(n_epoch):

        #reset game position back to top left
        qmaze.reset((0, 0))
        game_over = False
        optimizer.zero_grad()
        running_losses = []
        envstate = torch.from_numpy(qmaze.observe()).float()
        if device == "cuda":
            envstate = envstate.cuda()
        n_episodes = 0

        while not game_over:
            total_count += 1

            #
            valid_actions = qmaze.valid_actions()
            if not valid_actions:
                print('no valid')
                break
            qval = model(envstate)
            qval_ = qval.data.to('cpu').numpy()

            # greedy action selection strategy
            if random.random() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                action = np.argmax(qval_)

            # apply_action, get reward and new envstate
            new_envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win' or game_status == 'lose':
                # enc current epoch keeping track if the game was won
                win_history.append(game_status == 'win')
                game_over = True
            else:
                game_over = False

            # get new state
            new_envstate = torch.from_numpy(new_envstate).float()
            if device == "cuda":
                new_envstate = new_envstate.cuda()
            # store in the memory queue
            replay.append([envstate, action, reward, new_envstate, game_over])
            envstate = new_envstate
            n_episodes += 1
            if len(replay) > batch_size:
                #select the mini batch
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
                # get current predictions
                Q1 = model(envstate_batch)
                with torch.no_grad():
                    Q2 = model2(new_envstate_batch)
                # get y value of bellman's equ, the values we are trying to solve towards
                y = reward_batch + gamma * ((1 - game_over_batch) * torch.max(Q2, dim=1)[0])
                x = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = criterion(x, y.detach())
                optimizer.zero_grad()
                loss.backward()
                running_losses.append(loss.item())
                optimizer.step()

                # updated model 2 using sync frequency
                if total_count % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())

        print('epoch: ' + str(i + 1))
        print('loss: ' + str(np.mean(running_losses)))
        # can be multiple loss calculations depending on how many times the data was updated
        losses.append(np.mean(running_losses))
        if len(win_history) > min_epochs:
            # get the amount of wins the last (min_epochs)# of epochs
            win_rate = sum(win_history[-min_epochs:]) / min_epochs
        print('win rate: ' + str(win_rate))

        if win_rate == 1.0 and follows_path(qmaze, model2, optim_path, device):
            print("Solved to perfection")
            break

        if epsilon > 0.1:  # Decrements the epsilon value each epoch
            epsilon -= (epsilon_decay_factor / n_epoch)
    return model2
