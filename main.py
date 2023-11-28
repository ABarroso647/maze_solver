from Maze_Generation import maze



if __name__ == "__main__":
    m = maze()
    m.CreateMaze(10,10, loopPercent=50, saveMaze=True)
    m.run()
    m.maze_map()
