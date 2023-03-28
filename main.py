'''
This script is the main of the Score Four program. It initializes the two players, the game, and loop accross 
epoch (several consecutive games). It is the only script that needs to be manipulated to play the game,
selecting and initializing the players.
'''

from Game import Game
from GameState import GameState
from Player import PlayerRandom, PlayerHuman, PlayerSearchTreeAI, PlayerRLAI
from RLbasic import save_weights

# Create the players, the class defines the strategy

#player0 = PlayerHuman(ID = 0)
#player0 = PlayerRandom(ID = 0)
player0 = PlayerSearchTreeAI(ID = 0, depthMax = 3)
#player0 = PlayerRLAI(ID = 0, learn = False, weights_file = "RNDAI_vs_RLAI_500000ep_weights.pth")

#player1 = PlayerHuman(ID = 1)
#player1 = PlayerRandom(ID = 1)
player1 = PlayerSearchTreeAI(ID = 1, depthMax = 3)
#player1 = PlayerRLAI(ID = 1, learn = False, weights_file = "RLAI_SelfPlay_1000000ep_weights.pth")

numberOfGames = 10
gameLengths = [None] * numberOfGames
winners = [None] * numberOfGames

for i in range(numberOfGames):
    print(f'Game {i}')
    game = Game(player0 = player0, player1 = player1, isVerbose = False, gameState = GameState())
    game.run()
    #basic statistic collection on the game once it's ended
    gameLengths[i] = game.CurrentGameState.MoveCount
    winners[i] = game.CurrentGameState.getWinner()

print(f"Average game length : {sum(gameLengths)/len(gameLengths)} moves")
print(f"Player 0 won {len([x for x in winners if x == 0])} \nPlayer 1 won {len([x for x in winners if x == 1])}")

if isinstance(player1, PlayerRLAI):
    if player1.learn == True:
        file_path = f"{player0.name}_vs_{player1.name}_{numberOfGames}ep_weights.pth"
        save_weights(player1.model, file_path)

if isinstance(player0, PlayerRLAI):
    if player0.learn == True:
        file_path = f"{player0.name}_vs_{player1.name}_{numberOfGames}ep_weights.pth"
        save_weights(player0.model, file_path)
