import random
import time

#### Variables
START_POSITION = [0, 0]

# Enable/disable timing tests
ENABLE_TIMING = True


#### Creature
class Creature:
    def __init__(self, maze, brain, start_position, mutation_rate):
        self.position = start_position.copy()

        self.brain = brain  # list of desired direcitons at each move
        self.moves = 0  # number of moves taken
        self.dead = False  # if the creature has ran out of moves
        self.current_path = [tuple(start_position)]
        self.completed_maze = False
        self.fitness = 404

        self.maze = maze

        self.GOAL_POSITION = [len(maze) - 1, len(maze) - 1]
        self.MUTATION_RATE = mutation_rate  # mutate 15% of the moves

    def move(self):
        # perform moves until dead or completed maze
        while not self.dead and not self.completed_maze:

            valid_moves = self.get_valid_moves()

            if(len(valid_moves) == 1):
                # do the only valid move and dont use up a chromosome

                # update position and path
                self.position[0] += valid_moves[0][0]
                self.position[1] += valid_moves[0][1]
                self.current_path.append(tuple(self.position))

                if self.position == self.GOAL_POSITION:
                    self.completed_maze = True
                    break
            else:
                self.moves += 1
                if (self.moves > len(self.brain) - 1):
                    self.dead = True
                    self.moves = len(self.brain)
                    break

                if(len(valid_moves) == 0):
                    self.moves = len(self.brain)
                    self.dead = True
                    break

                # check if the next desired move is valid
                if not (self.brain[self.moves] not in valid_moves):
                    # update position and path
                    self.position[0] += self.brain[self.moves][0]
                    self.position[1] += self.brain[self.moves][1]
                    self.current_path.append(tuple(self.position))

                    if self.position == self.GOAL_POSITION:
                        self.completed_maze = True
                        break
                
    def calculate_fitness(self):
        # fitness is the number of moves taken to complete the maze or die
        if(self.completed_maze):
            self.fitness = len(self.current_path)
        else:
            self.fitness = self.moves
        # fitness is lower the better
        # fitness goes up if the creature is far from the goal
        self.fitness += abs(self.position[0] - self.GOAL_POSITION[0]) + abs(
            self.position[1] - self.GOAL_POSITION[1])
        
        # print(self.fitness)
        # print(self.current_path)

        # print(self.position, fitness, self.moves, self.completed_maze, self.dead, len(self.brain))

    def mutate(self):
        # randomly change some percent of the moves in the brain
        for i in range(int(len(self.brain) * self.MUTATION_RATE)):
            self.brain[random.randint(0, len(self.brain) - 1)] = random.choice([[0, 1], [1, 0], [-1, 0], [0, -1]])

    def get_valid_moves(self):
        valid_moves = []

        directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]

        for dx, dy in directions:
            newx = self.position[0] + dx
            newy = self.position[1] + dy

            if (newx < 0 or newy < 0 or newx >= len(self.maze) or newy >= len(self.maze[0])
                    or self.maze[newx][newy] != 0 or (newx, newy) in self.current_path):
                continue
            valid_moves.append([dx, dy])

        return valid_moves

def run_genetic_algorithm(maze, shortest_path_len, creature_lifespan, creature_count, mutation_rate, max_iterations):
    CREATURE_LIFESPAN = creature_lifespan
    CREATURE_COUNT = creature_count
    MAX_ITERATIONS = max_iterations

    # Known shortest path length
    known_shortest_path_len = shortest_path_len
    #print(f"Shortest path length: {known_shortest_path_len}")

    iterations = 0
    lowest_fitness = float('inf')  # lower is better
    shortest_path = []

    # Start, end times as well as figuring out how long it took to
    # A. Complete the maze and B. Find the shortest path.
    start = time.time()
    first_maze_completion = False
    shortest_path_completion = False
    shortest_path_length = 999999999
    shortest_path__len_no_revisit = 999999999  # Shortest path when revisited nodes are not counted
    end_time_first_maze_completion = time.time()
    end_time_shortest_path_completion = time.time()

    fittest_path = []
    fittest_path_fitness = 9999999999

    creatures = []

    while iterations < MAX_ITERATIONS:
        # Generate a bunch of creatures and move them around the maze

        if (iterations == 0):
            for i in range(CREATURE_COUNT):
                brains = []
                # generate a random brain for each creature
                possible_directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
                for i in range(CREATURE_LIFESPAN):
                    brains.append(random.choice(possible_directions))

                creatures.append(Creature(maze, brains, START_POSITION, mutation_rate))

        for creature in creatures:
            # Move each creature
            creature.move()
            creature.calculate_fitness()

            if(creature.fitness < fittest_path_fitness):
                fittest_path_fitness = creature.fitness
                fittest_path = creature.current_path.copy()

            # If the creature completes the maze, keep track of it if was the shortest path so far
            if creature.completed_maze:
                # Timing calculations
                if ENABLE_TIMING and not first_maze_completion:
                    first_maze_completion = True
                    end_time_first_maze_completion = time.time()
                    print("Time to find a complete path through the maze: ",
                          end_time_first_maze_completion - start)
                if len(creature.current_path) < shortest_path_length:
                    # Timing calculations
                    if (ENABLE_TIMING and not shortest_path_completion):
                        #and
                            #len(set(creature.current_path)) == known_shortest_path_len):
                        shortest_path_completion = True
                        end_time_shortest_path_completion = time.time()
                        print("Time to find the shortest path through the maze: ",
                            end_time_shortest_path_completion - start)
                    shortest_path_length = len(creature.current_path)
                    #shortest_path__len_no_revisit = len(set(creature.current_path))
                    shortest_path = creature.current_path

        # Sort the creatures by fitness
        creatures.sort(key=lambda x: x.fitness)

        # pick top 10% of creatures and mutate them
        survivors = creatures[:int(len(creatures) * 0.1)]

        # print info for the generation
        # print("Generation: ", iterations)
        # print("Creatures: ", len(creatures))
        # print(f"Shortest Path Length: {shortest_path_length} ")#({shortest_path__len_no_revisit} without revisits)
        # print("Shortest Path: ", shortest_path)
        # print("Fittest Creature This Generation Fitness: ", creatures[0].fitness)
        # print("Fittest Path Fitness: ", fittest_path_fitness)

        creatures = []

        for survivor in survivors:
            for i in range(10):
                creatures.append(Creature(maze, survivor.brain.copy(), START_POSITION, mutation_rate))
                creatures[-1].mutate()

        iterations += 1

    # print info for the generation
    print("Final results:")
    print(f"Shortest Path Length: {shortest_path_length} ") # ({shortest_path__len_no_revisit} without revisits)

    if ENABLE_TIMING:
        if first_maze_completion:
            print("Time to find a complete path through the maze: ",
                  end_time_first_maze_completion - start)

        if shortest_path_completion:
            print("Time to find the shortest path through the maze: ",
                  end_time_shortest_path_completion - start)

    # Plot the shortest path we calculated
    # plot_maze(maze, shortest_path)
    return fittest_path