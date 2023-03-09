from abc import ABC, abstractmethod
from GameState import GameState
import random

class Player(ABC):
    @abstractmethod
    def strategy(self, gameState : GameState) -> tuple: #return a move : a legal triplet of coordinates in the grid
        pass

class PlayerRandom(Player):
    def strategy(self, gameState: GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())
    
class PlayerHuman(Player) : 
    def strategy(self, gameState: GameState) -> tuple:
        print(f"Possible moves pick a number between 0 and {len(gameState.getPossibleMoves()) - 1} : \n {gameState.getPossibleMoves()}")
        return gameState.getPossibleMoves()[int(input())]
    
class PlayerAI(Player) :
    def strategy(self, gameState: GameState) -> tuple:
        #TODO : implement player strategy here
        pass
        