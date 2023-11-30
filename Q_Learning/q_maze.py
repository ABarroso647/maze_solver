import numpy as np
from numpy import float32

import Maze_Generation.gen as maze_gen
from typing import Optional
import matplotlib.pyplot as plt

# declared constants
VISITED = 0.8
AGENT_MARK = 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

ACTIONS = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}




class Qmaze:
    def __init__(self, maze: list[list[float]], initial_pos: tuple[int,int] = (0, 0)) -> None:
        self._maze = np.array(maze).astype(float32)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)  # bottom right = target
        zero_cells = np.where(self._maze == 0.0)
        # we assume maze is in 2d
        self.free_cells = list(zip(zero_cells[0], zero_cells[1]))
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 1.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not initial_pos in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(initial_pos)

    def reset(self, agent: tuple[int,int] ) -> None:
        self.agent = agent
        self.maze = np.copy(self._maze)
        row, col = agent
        self.maze[row, col] = AGENT_MARK
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action) -> None:
        cur_row, cur_col, cur_mode = self.state

        if self.maze[cur_row, cur_col] < 1.0:
            self.visited.add((cur_row, cur_col))  # mark as visited
        valid_actions = self.valid_actions()
        if not valid_actions:
            cur_mode = 'blocked'
        elif action in valid_actions:
            cur_mode = 'valid'
            if action == LEFT:
                cur_col -= 1
            elif action == UP:
                cur_row -= 1
            if action == RIGHT:
                cur_col += 1
            elif action == DOWN:
                cur_row += 1
        else:  # invalid action, no change in agent position
            cur_mode = 'invalid'

        self.state = (cur_row, cur_col, cur_mode)

    def get_reward(self) -> float:
        cur_row, cur_col, mode = self.state
        nrows, ncols = self.maze.shape
        if cur_row == nrows - 1 and cur_col == ncols - 1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (cur_row, cur_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            print("invalid")
            return -0.75
        else:
            # if mode == 'valid'
            return -0.04

    def act(self, action)-> (np.array, float, str):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self) -> np.array:
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        for i in range(nrows):
            for inc in range(ncols):
                if canvas[i,inc] < 0.5:
                    canvas[i, inc] = 0.0
        row, col, valid = self.state
        canvas[row, col] = AGENT_MARK
        return canvas

    def game_status(self)-> str:
        if self.total_reward < self.min_reward:
            return 'lose'
        cur_row, cur_col, mode = self.state
        nrows, ncols = self.maze.shape
        if cur_row == nrows - 1 and cur_col == ncols - 1:
            return 'win'

        return 'not over'

    def valid_actions(self, cell: Optional[tuple[int,int]]=None)-> list[int]:
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = list(ACTIONS.keys())
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self.maze[row - 1, col] == 1.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 1.0:
            actions.remove(3)

        if col > 0 and self.maze[row, col - 1] == 1.0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 1.0:
            actions.remove(2)

        return actions

    def path(self) -> list[tuple[int,int]]:
        visited = self.visited
        cur_row, cur_col, _ = self.state
        visited.add((cur_row, cur_col))
        return list(visited)



def show(qmaze: Qmaze):
    plt.grid()
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.4
    cur_row, cur_col, _ = qmaze.state
    canvas[cur_row, cur_col] = 0.7  # agent cell
    canvas[nrows - 1, ncols - 1] = 0.1  # final cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray_r')
    plt.show()
    return img



if __name__ == '__main__':
    print(len(ACTIONS))
    qmaze = Qmaze(maze_gen.maze)
    canvas, reward, game_over = qmaze.act(DOWN)
    qmaze.act(RIGHT)
    qmaze.act(DOWN)
    qmaze.act(DOWN)
    qmaze.act(DOWN)
    qmaze.act(DOWN)
    print("reward=", reward)
    show(qmaze)