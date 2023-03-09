# Score Four 
## Abstract game
For a rough information page on the game itself, check out [its Wikipedia page](https://en.wikipedia.org/wiki/Score_Four).
## Implementation
### GameState
This file and class contains most of the game logic. Also contains the parameters for the grid size and the win condition size (if you want to play Score 5). Not intended to be modified except for these parameters.
### Players
A player is required to implement a game strategy i.e return a legal move from a given gamestate. PlayerRandom and PlayerHuman give some examples of implementation for this class.
### main.py
Define the players to use then launch as many games of Score Four as you want and check your stats from here .
For reference about 3000 games per second can be completed with PlayerRandom and a standard 4 Size.
