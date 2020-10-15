
## Developing a fully convolutional Minesweeper model

This project is about the development of a neural network to play Minesweeper. This involved the following steps:
* Implementation of the logic of Minesweeper
    * Once in Python, including a minefield editor, a GUI, and model integration
    * And a threaded C++ implementation to accelerate generation of training data
* Generation of a set of unbiased state-action pairs for training a model
* Development of methods for training and evaluating models

The result was a model that makes safe moves 99.6% of the time, wins 41.1% of games played, and makes 178 consecutive safe moves on average on expert puzzles (16x30 grid with 99 mines), provided the random move at the beginning is safe. These numbers were obtained by averaging the results of 3000 randomly initialized expert puzzles.

Check this out! The model playing in the gif and the associated code are available in src/

![Minesweeper Model Demo :D](model_pwns_expert_minefield.gif)

## Getting Started

Before getting into detail on how all this works,  

## Introduction/Overview

### Problem Definition

### Technical Considerations

### Concepts

## Methods/Results:

### Step 1: 
