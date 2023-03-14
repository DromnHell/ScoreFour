from abc import ABC, abstractmethod
from GameState import GameState
import random
import math
import copy

current_IDs = list()

class Player(ABC):
    
    def __init__(self, ID) -> None:
        if ID not in (0, 1):
            print("The ID of the players has to be be 0 or 1.")
            quit()
        elif ID in current_IDs:
            print("The two players must have a different ID.")
            quit()
        elif not current_IDs and ID == 1:
            print("The first player must has the ID 0.")
            quit()
        else:
            current_IDs.append(ID)
            self.ID = ID

    @abstractmethod
    def strategy(self, gameState : GameState) -> tuple: #return a move : a legal triplet of coordinates in the grid
        pass

class PlayerRandom(Player):

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
    
    def strategy(self, gameState: GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())
    
class PlayerHuman(Player) : 

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)

    def strategy(self, gameState: GameState) -> tuple:
        print(f"Possible moves pick a number between 0 and {len(gameState.getPossibleMoves()) - 1} : \n {gameState.getPossibleMoves()}")
        return gameState.getPossibleMoves()[int(input())]
        
class PlayerAI1(Player) :

    def __init__(self, ID, evaluation, depth_limit) -> None:
        Player.__init__(self, ID)
        self.file_AI1 = open("file_AI1.txt", "w") 
        self.depth_limit = depth_limit
        self.eval_type = evaluation
        if self.eval_type not in ("move_quality", "grid_quality"):
            print("The AI know to type of évaluation : 'move_quality' and 'grid_quality'")
            quit()
    
    def grid_quality_score(self, gameState: GameState) -> int:
        grid = gameState.Grid
        if self.ID == 0:
            player = 0
            other_player = 1
        else:
            player = 1
            other_player = 0
        score = 0
        # Check if the central square of the first layer contains a player token
        if grid[0][2][1] == player or grid[0][1][1] == player or grid[0][1][2] == player or grid[0][2][2] == player:
            score += 5
        # Check the horizontal alignments in layers
        for x in range(4):
            for y in range(4):
                row = [grid[x][y][0], grid[x][y][1], grid[x][y][2], grid[x][y][3]]
                if row.count(player) == 4:
                    score += 1000
                elif row.count(player) == 3 and row.count(None) == 1:
                    score += 10
                elif row.count(player) == 2 and row.count(None) == 2:
                    score += 2
                elif row.count(other_player) == 2 and row.count(None) == 2:
                    score -= 2
                elif row.count(other_player) == 3 and row.count(None) == 1:
                    score -= 10
                elif row.count(other_player) == 4:
                    score -= 1000
        # Check the vertical alignments in layers
        for x in range(4):
            for z in range(4):
                col = [grid[x][0][z], grid[x][1][z], grid[x][2][z], grid[x][3][z]]
                if col.count(player) == 4:
                    score += 10000
                elif col.count(player) == 3 and col.count(None) == 1:
                    score += 10
                elif col.count(player) == 2 and col.count(None) == 2:
                    score += 2
                elif col.count(other_player) == 2 and col.count(None) == 2:
                    score -= 2
                elif col.count(other_player) == 3 and col.count(None) == 1:
                    score -= 10
                elif col.count(other_player) == 4:
                    score -= 10000
        # Check the diagonal alignments in layers
        for x in range(4):
            diag1 = [grid[x][0][0], grid[x][1][1], grid[x][2][2], grid[x][3][3]]
            diag2 = [grid[x][3][0], grid[x][2][1], grid[x][1][2], grid[x][0][3]]
            if diag1.count(player) or  diag2.count(player) == 4:
                score += 1000
            elif diag1.count(player) == 3 and diag1.count(None) == 1:
                score += 10
            elif diag2.count(player) == 3 and diag2.count(None) == 1:
                score += 10
            elif diag1.count(player) == 2 and diag1.count(None) == 2:
                score += 2
            elif diag2.count(player) == 2 and diag2.count(None) == 2:
                score += 2
            elif diag1.count(other_player) == 2 and diag1.count(None) == 2:
                score -= 2
            elif diag2.count(other_player) == 2 and diag2.count(None) == 2:
                score -= 2
            elif diag1.count(other_player) == 3 and diag1.count(None) == 1:
                score -= 10
            elif diag2.count(other_player) == 3 and diag2.count(None) == 1:
                score -= 10
            elif diag1.count(other_player) == 4 or diag2.count(other_player):
                score -= 1000
        # Check the "pillars" alignments
        for y in range(4):
            for z in range(4):
                pillar = [grid[0][y][z], grid[1][y][z], grid[2][y][z], grid[3][y][z]]
                if pillar.count(player) == 4:
                    score += 1000
                elif pillar.count(player) == 3 and pillar.count(None) == 1:
                    score += 10
                elif pillar.count(player) == 2 and pillar.count(None) == 2:
                    score += 2
                elif pillar.count(other_player) == 2 and pillar.count(None) == 2:
                    score -= 1
                elif pillar.count(other_player) == 3 and pillar.count(None) == 1:
                    score -= 3
                elif pillar.count(other_player) == 4:
                    score -= 1000
        # Check the diagonal alignments betwen layers
        diag1 = [grid[0][0][0], grid[1][1][1], grid[2][2][2], grid[3][3][3]]
        diag2 = [grid[0][3][0], grid[1][2][1], grid[2][1][2], grid[3][0][3]]
        if diag1.count(player) == 4 or diag1.count(player) == 4:
            score += 1000
        elif diag1.count(player) == 3 and diag1.count(None) == 1:
            score += 10
        elif diag2.count(player) == 3 and diag2.count(None) == 1:
            score += 10
        elif diag1.count(player) == 2 and diag1.count(None) == 2:
            score += 2
        elif diag2.count(player) == 2 and diag2.count(None) == 2:
            score += 2
        elif diag1.count(other_player) == 2 and diag1.count(None) == 2:
            score -= 2
        elif diag2.count(other_player) == 2 and diag2.count(None) == 2:
            score -= 2
        elif diag1.count(other_player) == 3 and diag1.count(None) == 1:
            score -= 10
        elif diag2.count(other_player) == 3 and diag2.count(None) == 1:
            score -= 10
        elif diag1.count(other_player) == 4 or diag2.count(other_player):
            score -= 1000
        return score
    
    def move_quality_score(self, gameState: GameState) -> int:
        score = 0
        if gameState.getWinner() != None:
            score += 100
        self.file_AI1.write(f'Score = {score}\n')
        return score
    
    def evaluation(self, gameState: GameState) -> int:
        if self.eval_type == "move_quality":
            score = self.move_quality_score(gameState)
        else:
            score = self.grid_quality_score(gameState)
        return score

    def AlphaBetaPruning(self, gameState: GameState, depth, alpha, beta, maximizingPlayer) -> int:
        self.file_AI1.write(f'Depth = {depth}\n')
        if maximizingPlayer == True:
            self.file_AI1.write(f'Maximizing player turn !\n')
        else:
            self.file_AI1.write(f'Minimizing player turn !\n')
        # Terminating condition
        if depth == 0 or gameState.checkEnd():
            return self.evaluation(gameState)
        # Maximizing player block
        if maximizingPlayer:
            bestValue = -math.inf
            # Recur on all children
            for move in gameState.getPossibleMoves():
                self.file_AI1.write('---------------------------\n')
                self.file_AI1.write(f"Move : {move}\n")
                new_gameState = copy.deepcopy(gameState)
                new_gameState.playLegalMove(move)
                value = self.AlphaBetaPruning(new_gameState, depth - 1, alpha, beta, False)
                self.file_AI1.write(f'Value = {value}\n')
                bestValue = max(bestValue, value)
                self.file_AI1.write(f'Best value = {bestValue}\n')
                self.file_AI1.write(f'Alpha before = {alpha}\n')
                alpha = max(alpha, bestValue)
                self.file_AI1.write(f'Alpha after = {alpha}\n')
                self.file_AI1.write(f'Beta = {beta}\n')
                # Alpha Beta Pruning
                if beta <= alpha:
                    self.file_AI1.write(f'BREAK\n')
                    break
                self.file_AI1.write('---------------------------\n\n')
            return bestValue
        # Minimizing player block
        else:
            bestValue = math.inf
            # Recur on all children
            for move in gameState.getPossibleMoves():
                self.file_AI1.write('---------------------------\n')
                self.file_AI1.write(f"Move : {move}\n")
                new_gameState = copy.deepcopy(gameState)
                new_gameState.playLegalMove(move)
                value = self.AlphaBetaPruning(new_gameState, depth - 1, alpha, beta, True)
                self.file_AI1.write(f'Value = {value}\n')
                bestValue = min(bestValue, value)
                self.file_AI1.write(f'Best value = {bestValue}\n')
                self.file_AI1.write(f'Beta before = {beta}\n')
                beta = min(beta, bestValue)
                self.file_AI1.write(f'Beta after = {beta}\n')
                self.file_AI1.write(f'Alpha = {alpha}\n')
                # Alpha Beta Pruning
                if beta <= alpha:
                    self.file_AI1.write(f'BREAK\n')
                    break
                self.file_AI1.write('---------------------------\n\n')
            return bestValue

    def strategy(self, gameState: GameState) -> tuple:

        best_move = None

        for depth in range(1, self.depth_limit + 1):

            self.file_AI1.write('---------------------------\n')
            self.file_AI1.write('---------------------------\n')
            self.file_AI1.write('---------------------------\n')
            self.file_AI1.write(f'Depth = {depth - 1}\n')

            for move in gameState.getPossibleMoves():

                self.file_AI1.write('---------------------------\n')
                self.file_AI1.write('---------------------------\n')

                self.file_AI1.write(f"Move : {move}\n")

                new_gameState = copy.deepcopy(gameState)
                new_gameState.playLegalMove(move)

                value = self.AlphaBetaPruning(new_gameState, depth - 1, -math.inf, math.inf, True)

                self.file_AI1.write('---------------------------\n')
                self.file_AI1.write('---------------------------\n\n')

                if best_move is None or value > best_move[1]:
                    best_move = (move, value)

            self.file_AI1.write('---------------------------\n')
            self.file_AI1.write('---------------------------\n')
            self.file_AI1.write('---------------------------\n\n')
                    
        return best_move[0]
