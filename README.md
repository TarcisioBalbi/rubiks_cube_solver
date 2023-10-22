# Rubik's Cube Solver

This repository contains an application that implements a 2x2 Rubik's Cube and trains a random forest classifier to solve it.


# Requirements
Make sure you have the following requirements installed in your development environment:

- Python 3.8
- Required libraries can be installed using pip install -r requirements.txt

# Run
In order to run this application, follow these steps:
1 - Run `00_generate_data.py`.
2 - Run `01_train_model.py`.
3 -Run the app file `app.py`

Be aware that steps 1 and 2 will try to save files in a `data` folder, so make sure to create it before.

# Data generation
This application generates data by starting from a solved cube, scrambling it and then reversing the order. For each movement, the current state and the movement are saved.
Each movement is represented by a string following the pattern `side,direction', where:
- side: 0 = front, 1 = back, 2 = upper, 3 = bottom, 4 = right, 5 = left
- directions 0 = 90° clockwise , 2 = 90° counter-clockwise
- 
The parameters for this step are:
- n_runs: number of times the process is repeated with a new cube.
- n_moves: number of scrambling move made in each cube.
- random_state (optional): set it if you want consistent results.

# Model training
The model is a simple random forest classifier, optimized by grid search, used as proof of concept. More complex models, specially ones with memory, can probably achieve better results.

# Application
The `play` function takes:
- cube: vector representing the initial cube positions.
- model: model used to solve it.
- n_scramble_moves: number of random movements applied to the initial cube.
- random_state (optional): set it if you want consistent results
