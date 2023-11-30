from Maze_Generation.gen import plot_maze, find_shortest_path
import time
import torch
import yaml

from Q_Learning.q_model import Qmaze, train_model, play_game
from Maze_Generation.gen import MAZE, BIG_MAZE, find_shortest_path
from Ant_Colony_Optimization.aco import run_ant_colony
from Genetic_Algo.genetic import run_genetic_algorithm

if __name__ == '__main__':
    with open('./config/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    maze = MAZE
    if config['maze_size'] == 'big':
        maze = BIG_MAZE

    shortest_path = find_shortest_path(maze)
    plot_maze(maze, shortest_path)

    if config['search_algorithm'] == 'q_learning':
        # hyperparameters
        n_epoch = config['q_learning']['n_epoch']
        lr = config['q_learning']['lr']
        epsilon = config['q_learning']['epsilon']
        epsilon_decay_factor = config['q_learning']['epsilon_decay_factor']
        mem_size = config['q_learning']['mem_size']
        batch_size = config['q_learning']['batch_size']
        gamma = config['q_learning']['gamma']
        sync_freq = config['q_learning']['sync_freq']
        min_epochs = config['q_learning']['min_epochs']

        q_maze = Qmaze(maze)
        t0 = time.time()
        model = train_model(q_maze, shortest_path, n_epoch=n_epoch, lr=lr, epsilon=epsilon,
                            epsilon_decay_factor=epsilon_decay_factor,
                            mem_size=mem_size, batch_size=batch_size, gamma=gamma, sync_freq=sync_freq,
                            min_epochs=min_epochs)
        t1 = time.time()
        total = t1 - t0
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        play_game(q_maze, model, device)
        print(f"Total time to converge: {total}")

        plot_maze(maze, q_maze.path())
    elif config['search_algorithm'] == 'ACO':
        # hyperparameters
        max_iterations = config['ACO']['max_iterations']
        ant_count = config['ACO']['ant_count']
        max_moves = config['ACO']['max_moves']

        max_iter = config['ACO']['max_total_iterations']

        print(f"Shortest path length: {len(shortest_path)}")

        path = run_ant_colony(maze, len(shortest_path), max_iterations, ant_count, max_moves)
        # plot final iteration
        plot_maze(maze, path)
    elif config['search_algorithm'] == 'genetic':
        # hyperparameters
        max_iterations = config['genetic']['max_iterations']
        creature_lifespan = config['genetic']['creature_lifespan']
        creature_count = config['genetic']['creature_count']
        mutation_rate = config['genetic']['mutation_rate']

        path = run_genetic_algorithm(maze, len(shortest_path), creature_lifespan, creature_count, mutation_rate, max_iterations)
        plot_maze(maze, path)
