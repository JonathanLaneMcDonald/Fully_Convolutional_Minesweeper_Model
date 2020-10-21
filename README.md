
## Developing a fully convolutional Minesweeper model

This project is about the development of a neural network to play Minesweeper. This involved the following steps:
* Implementation of the logic of Minesweeper
    * Once in Python, including a minefield editor, a GUI, and model integration
    * And a threaded C++ implementation to accelerate generation of training data
* Generation of a set of unbiased state-action pairs for training a model
* Development of methods for training and evaluating models

The result was a model that makes safe moves 99.6% of the time, wins 41.1% of games played, and makes 178 consecutive safe moves on average on expert puzzles (16x30 grid with 99 mines), provided the random move at the beginning is safe. These numbers were obtained by averaging the results of 3000 randomly initialized expert puzzles.

Check this out! The model playing in the gif and the associated code are available in src/

![Minesweeper Model Demo :D](model_pwns_expert_minefield.gif~)

## Getting Started

Before getting into detail on how all this works, a few details on how to proceed with compiling and running everything because I haven't made it particularly user-friendly yet :D

1) Compile and run the dataset generator, head to the src/ directory and run:
    ```
    g++ minesweeper_main.cpp -o minesweeper -lpthread -O3
    ```
   
    Once compiled, you'll build your datasets using the following parameter scheme

    ```
    ./minesweeper [rows] [cols] [mines] [desired number of samples] [execution threads]
    ```

    As an example, you might run the two lines below to generate your training and validation sets using 8 threads. 

    ```
    ./minesweeper 16 30 99 1000000 8 > training
    ./minesweeper 16 30 99 100000 8 > validation
    ```

    * Note: that I wrote this on a linux machine and it compiles and runs fine on windows, but threading doesn't work yet in windows, so it seems fastest to run in 1 thread for now unless you're on a linux machine. I'll fix that at some point.
    * Note: defaults for grid size and number of mines are defined in common.py, so make sure they match the dimensions of the dataset you generate.

2) Train a model on the generated datasets

    ```
    python train.py training validation
    ```

    This will take a while, depending on your hardware, but trained models will start popping out. You may need to adjust the batch size depending on how much memory you have available. There will also be a progress report after each training round so you can see how everything's going.

3) Evaluate a trained model

    ```
    python evaluate.py "minesweeper model 16x30x99"
    ```
    
    Running evaluate.py with a model will give you some cryptic output about win rates and safe move rates and likelihood that a random sampler would get the same results and stuff like that. It's pretty cool :D

4) Play a game in a GUI with model integration

    To play without model integration, just run the play script
    
    ```
    python play.py
    ```
    
    To play *with* model integration, include the path to a trained model
    
    ```
    python play.py "minesweeper model 16x30x99"
    ```
    
    * Note: models only work on the grid size they were trained on (for now)

## Introduction/Overview

### Problem Definition

The goal is to train a neural network to predict safe moves in games of minesweeper.

### Technical Considerations

**Data Representation**

Minesweeper is a 2D game where each cell can be mined, flagged, visited, or unvisited, but it's the layer that specifies the number of nearby mines that contains the information. 

Here's my thought process: 
* Discrete numbers can be represented as binary features in separate channels in the input
* Unvisited cells can be represented by not setting any values for unvisited positions
* The numbers 0-8 will cover all likely scenarios, but I'll throw in a channel for 9 ;)

The result is a binary input with shape [None][r][c][10] where r=rows, c=cols, and 10 is the number of channels (and None is the sample dimension).

**Training Objective**

So what am I trying to predict, anyway? This took some trial and error because concepts like bias started seeping in. Anyway, the major approaches were as follows:
1) There's a false equivalence here, but my first attempt was modeled after computer Go models like AlphaGo where I'd train the model to select *a* good move and then apply a softmax and use categorical crossentropy as my loss (even though Minesweeper isn't "that kind" of game).
2) The successful approach was to have the model predict all safe moves in one go. So now we're using a sigmoid activation instead of a softmax and we're using a binary crossentropy loss instead of a categorical one. That's basically the current state of things.


### Concepts

**Bias**



## Methods/Results:

### Step 1: Acquire Training Data

### Step 2: Designing the Model

### Step 3: Training the Model

### Step 4: Evaluating the Model

### Final Stage: Playing some vidja games

