import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from matplotlib.colors import ListedColormap

# DO NOT MODIFY MAZE
MAZE = [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
]

BIG_MAZE = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]


# DO NOT MODIFY MAZE

# Find the shortest path 
# To use, call the function and prove it the maze above, start and end, which are tuples
# (0,0) and (9,9) respectively

# Function will return a list of lists of the coordinates of the shortest path by BFS
# eg. [[0, 0], [1, 0], [1, 1]...]

# Use this function to test against your own algo to make sure it can find the shortest path
def find_shortest_path(maze):
    maze = np.array(maze)
    start = (0, 0)
    end = (maze.shape[0]-1, maze.shape[1]-1)

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    size = len(maze)
    visited = [[False] * size for _ in range(size)]
    prev = [[None] * size for _ in range(size)]

    q = Queue()
    q.put(start)
    visited[start[0]][start[1]] = True

    while not q.empty():
        x, y = q.get()
        if (x, y) == end:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny] and maze[nx][ny] == 0:
                q.put((nx, ny))
                visited[nx][ny] = True
                prev[nx][ny] = (x, y)

    path = []
    if prev[end[0]][end[1]] is not None:
        at = end
        while at:
            path.append(at)
            at = prev[at[0]][at[1]]
        path.reverse()

    return path

# Function plots the maze and provided (shortest) path
# Function takes in the maze above which should not be modified
# Function takes in the path you provide which will be a list of list of coordinates
# eg. [[0, 0], [1, 0], [1, 1]...]

# Function will pop up a matplotlib diagram when exectuted
def plot_maze(maze, path=None):
    maze = np.array(maze)
    if path is not None:
        maze[0][0] = 2
        for x, y in path:
            maze[x][y] = 2

    cmap = ListedColormap(['white', 'black', 'green'])

    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap=cmap, interpolation='none')

    for x in range(len(maze[0]) + 1):
        plt.axhline(y=x - 0.5, color='black', linestyle='-', linewidth=2)
        plt.axvline(x=x - 0.5, color='black', linestyle='-', linewidth=2)

    plt.xticks([x - 0.5 for x in range(1, len(maze[0]))], [])
    plt.yticks([y - 0.5 for y in range(1, len(maze))], [])

    plt.tick_params(axis=u'both', which=u'both', length=0)

    plt.arrow(0 - 0.4, 0, 0.4, 0, fc='red', ec='red', head_width=0.3, head_length=0.3)
    plt.arrow(maze.shape[1] - 1 - 0.4, maze.shape[0] - 1, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)

    plt.show()

'''
To run the code above, first:

Install matploblib by running pip3 install -r requirements.txt

then

import sys
sys.path.append('../Maze_Generation/')
from gen import find_shortest_path, plot_maze, maze

then

Then execute path = find_shortest_path(maze, (0,0), (9,9)) to get the BFS shortest path
Then find your shortest path and save it to a variable
Then execute plot_maze(maze, your_shortest_path) to see your path plotted on the maze
Compare against the BFS shortest path to ensure it works
'''
