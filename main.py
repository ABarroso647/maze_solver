from Maze_Generation.gen import plot_maze, find_shortest_path
import numpy as np
import time
import torch

import sys
from typing import Callable
import yaml


from Q_Learning.q_model import Qmaze, train_model, play_game
from Maze_Generation.gen import MAZE, BIG_MAZE, find_shortest_path
from Ant_Colony_Optimization.aco import run_ant_colony

if __name__ == '__main__':
    with open('./config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    maze = MAZE
    if config['maze_size'] == 'big':
        maze = BIG_MAZE

    if config['search_algorithm'] == 'q_learning':
        # hyperparameters
        n_epoch = config['q_learning']['n_epoch']
        lr = config['q_learning']['lr']
        epsilon = config['q_learning']['epsilon']
        mem_size = config['q_learning']['mem_size']
        batch_size = config['q_learning']['batch_size']
        gamma = config['q_learning']['gamma']
        sync_freq = config['q_learning']['sync_freq']
        h_size = config['q_learning']['h_size']

        test = find_shortest_path(maze)
        q_maze = Qmaze(maze)
        t0 = time.time()
        model = train_model(q_maze, test, n_epoch=n_epoch, lr=lr, epsilon=epsilon, mem_size=mem_size, batch_size=batch_size, gamma=gamma, sync_freq=sync_freq, h_size=h_size)
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        play_game(q_maze, model, device)
        t1 = time.time()
        total = t1-t0
        print(f"Total time to converge: {total}")

        plot_maze(maze, q_maze.path())
