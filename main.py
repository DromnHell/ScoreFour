from Game import Game
from GameState import GameState
from Player import PlayerRandom, PlayerHuman, PlayerAI

# Create the players, the class defines the strategy
player0 = PlayerRandom()
player1 = PlayerRandom()
# use PlayerAI for your own implementation (for player0 or player1)
#player1 = PlayerAI()

numberOfGames = 5000
gameLengths = [None] * numberOfGames
winners = [None] * numberOfGames

for i in range(numberOfGames):
    game = Game(player0= player0, player1= player1, isVerbose= False, gameState=GameState())
    game.run()
    #basic statistic collection on the game once it's ended
    gameLengths[i] = game.CurrentGameState.MoveCount
    winners[i] = game.CurrentGameState.getWinner()

print(f"Average game length : {sum(gameLengths)/len(gameLengths)} moves")
print(f"Player 0 won {len([x for x in winners if x == 0])} \nPlayer 1 won {len([x for x in winners if x == 1])}")
