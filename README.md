# maze_solver
This repo aims to solve simple mazes with multiple different algorithms and compare the results

## Algorithms

- Deep Q Network
- Genetic Algorithm
- Ant Colony Optimization

## Usage

Edit [config.yml](./config/config.yml) to change the desired algorithm and hyperparameters.

Install dependencies:
```bash
python3 -m venv ./venv
. ./venv/bin/activate
pip3 install -r requirements.txt
```

Run the algorithm:
```bash
python3 main.py
```
