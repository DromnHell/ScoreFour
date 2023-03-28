# Score Four

## Source
This repository is a fork of the https://github.com/Latylus/ScoreFour repository maintained by Pierre Lataillade.

## Abstract game
For a rough information page on the game itself, check out [its Wikipedia page](https://en.wikipedia.org/wiki/Score_Four).

## Implementation

### main.py
Define the players to use then launch as many games of Score Four as you want and check your stats from here.
For reference about 3000 games per second can be completed with PlayerRandom and a standard 4 Size.

#### Modifications and additions to the original file
* Adding the call to save_weights() function to save lhe weights of the RL AI.

### Game
Define the class of the game and the principal loop game.

#### Modifications and additions to the original file
* Adding the call to receive_last_feedback() function to send the wining action of winner player to the looser one. Used only by the RL AI.
* Adding a piece of code that gradually decreases epsilon. Used only by the RL AI.

### Players
A player is required to implement a game strategy i.e return a legal move from a given gamestate. PlayerRandom and PlayerHuman give some examples of implementation for this class.

#### Modifications and additions to the original file
* Adding conditions in the constructor of the Player(ABC) class to better manage the initialization of players IDs.
* Change the system for entering an imput for the human player, so that if a wrong action is entered, it can be restarted without crash the game.
* Adding the classes of the players PlayerSearchTreeAI (fully functionnal) and PlayerRLAI (not fully functional) and their methods.

### GameState
This file and class contains most of the game logic. Also contains the parameters for the grid size and the win condition size (if you want to play Score 5). Not intended to be modified except for these parameters.

#### Modifications and additions to the original file
* Adding a copy() function to copy a game state.
* Modification of the getPossibleMoves() actions to save the original actions index.

### RLbasic (new file)
This file contains the class of a neural network (NN) and functions that allow the player "PlayerRLAI" to learn and play Score Four.
The train_self_play() function also allows  the NN to train the itself without going through the game loop of the "Game.py" script, but it not fully functional at the moment.

