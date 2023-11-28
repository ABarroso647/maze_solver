import sys
sys.path.append('../Maze_Generation/')
from gen import find_shortest_path, plot_maze, maze
import random
import time

# Number of steps each ant can take
ANT_MAX_MOVES_HYPER_PARAM = 30
# Number of times ants are released into maze (generations)
ANT_COLONY_MAX_ITERATIONS = 10
# Numbers of ants released each iteration
ANT_COUNT = 10
# The end of the maze (bottom right corner)
GOAL_POSITION = [9,9]
# Max number of times to run the entire algorithm for timings
MAX_TOTAL_ITERATIONS = 10
# Enable/disable timing tests
ENABLE_TIMING = False
# Known shortest path length
KNOWN_SHORTEST_PATH_LENGTH = 19

class Ant:
    def __init__(self, maze, pheremones):
        self.position = [0,0]
        self.current_path = [[0,0]]
        self.moves = 0
        self.maze = maze
        self.pheremones = pheremones
        self.completed_maze = False
        self.dead_end = False
        self.visited = set((0,0))


    def move(self):
        while self.moves < ANT_MAX_MOVES_HYPER_PARAM:
            probabilities = []
            all_pheremones = 0

            # If there are no valid moves (eg. dead end) the ant stops moving and marks itself
            # as having reached a dead end
            valid_moves = self.get_valid_moves()
            if len(valid_moves) == 0:
                self.dead_end = True
                break

            # Get all surrounding pheremones of a position
            # Used in the calculation of a positions pheremone
            for dx, dy in valid_moves:
                newx = self.position[0] + dx
                newy = self.position[1] + dy
                all_pheremones += self.pheremones[newx][newy]

            # Calculate the weighted probability of moving in any valid direction
            # Uses pheremones to calculate probability and appends to a list of probabilities for each respective position
            for dx, dy in valid_moves:
                newx = self.position[0] + dx
                newy = self.position[1] + dy

                move_probability = self.pheremones[newx][newy]/all_pheremones
                probabilities.append(move_probability)

            # Choose a random move based on the weighted probability previously calculated
            selected_move = random.choices(valid_moves, weights=probabilities, k=1)[0]

            # Update position and path so far and add to visited set to avoid backtracking
            self.position[0] += selected_move[0]
            self.position[1] += selected_move[1]
            self.current_path.append(list(self.position))
            self.moves += 1
            self.visited.add((self.position[0], self.position[1]))

            # If reached the end, ant stops moving and marks itself as completed_maze
            if self.position == GOAL_POSITION: 
                self.completed_maze = True
                break

    # Calculates which direction an ant is able to move
    # Return value: list of list of directions an ant can move
    # eg: [[0,1], [1,0], [-1,0]]
    def get_valid_moves(self):
        valid_moves = []

        directions = [[0,1], [1,0], [-1,0], [0,-1]]

        for dx, dy in directions:
            newx = self.position[0] + dx
            newy = self.position[1] + dy

            if (newx < 0 or newy < 0 or newx >= len(self.maze) or newy >= len(self.maze[0]) 
                or self.maze[newx][newy] != 0 or (newx,newy) in self.visited):
                continue
            valid_moves.append([dx,dy])        

        return valid_moves

class Maze:

    def __init__(self, maze):
        self.maze = maze
        # Initialize a matrix of pheremones to its respective position in the original matrix
        self.pheremone_matrix =  [[1 for _ in range(len(self.maze[0]))] for _ in range(len(self.maze))]
        self.evaporation_rate = 0.5
        self.pheremone_amount = 1

    def update_pheremones(self, ants):
        # If an ant has completed the maze boost the pheremones of that path
        # If an ant reaches a dead end, scale down pheremones of that path
        for ant in ants:
            pheremone_amount = self.pheremone_amount
            if ant.completed_maze:
                pheremone_amount = 1.1
            if ant.dead_end:
                pheremone_amount = 0.8

            for x,y in ant.current_path:
                self.pheremone_matrix[x][y] *= pheremone_amount

        # Evaporate all pheremones based on evaporation rate
        for _,row in enumerate(self.pheremone_matrix):
            for _, pheremone in enumerate(row):
                pheremone *= (1-self.evaporation_rate)

def run_ant_colony():
    iterations = 0
    shortest_path_length = float('inf')
    shortest_path = []
    final_maze = Maze(maze)
    
    # Start, end times as well as figuring out how long it took to
    # A. Complete the maze and B. Find the shortest path.
    start = time.time()
    first_maze_completion = False
    shortest_path_completion = False
    
    while iterations < ANT_COLONY_MAX_ITERATIONS:
        # Generate an ANT_COUNT number of ants with the originial maze
        # and an updated pheremone matrix
        ants = [Ant(final_maze.maze, final_maze.pheremone_matrix) for _ in range(ANT_COUNT)]
        for ant in ants:
            # Move each ant
            ant.move()
            # If the ant completes the maze, keep track of it if was the shortest path so far
            if ant.completed_maze:
                # Timing calculations
                if ENABLE_TIMING and not first_maze_completion:
                    first_maze_completion = True
                    end_time_first_maze_completion = time.time()
                    print("Time to find a complete path through the maze: ", 
                          end_time_first_maze_completion - start)

                if len(ant.current_path) < shortest_path_length:
                    # Timing calculations
                    if (ENABLE_TIMING and not shortest_path_completion and
                        len(ant.current_path) == KNOWN_SHORTEST_PATH_LENGTH):
                        shortest_path_completion = True
                        end_time_shortest_path_completion = time.time()
                        print("Time to find the shortest path through the maze: ",
                              end_time_shortest_path_completion - start)
                    shortest_path_length = len(ant.current_path)
                    shortest_path = ant.current_path
        
        # Update pheremones after all ants have moved
        final_maze.update_pheremones(ants)
        iterations += 1

    # Plot the shortest path we calculated
    if not ENABLE_TIMING:
        plot_maze(final_maze.maze, shortest_path)

    
if __name__ == "__main__":
    if ENABLE_TIMING:
        count = 0
        while count < MAX_TOTAL_ITERATIONS:
            count += 1
            run_ant_colony()
    else:
        run_ant_colony()

