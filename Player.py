from abc import ABC, abstractmethod
import GameState
import random
import math
import numpy as np

current_IDs = list()

class Player(ABC):
    
    def __init__(self, ID) -> None:
        if ID not in (0, 1):
            print("Error : the ID of the players has to be be 0 or 1.")
            quit()
        elif ID in current_IDs:
            print("Error :the two players must have a different ID.")
        elif not current_IDs and ID == 1:
            print("Error : the first player must have the ID 0.")
            quit()
        else:
            current_IDs.append(ID)
            self.ID = ID

    @abstractmethod
    def strategy(self, gameState : GameState.GameState) -> tuple: #return a move : a legal triplet of coordinates in the grid
        pass

class PlayerRandom(Player):

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
    
    def strategy(self, gameState: GameState.GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())
    
class PlayerHuman(Player) : 

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)

    def strategy(self, gameState: GameState.GameState) -> tuple:
        possible_moves = gameState.getPossibleMoves()
        nb_possible_moves = len(possible_moves)
        valid_input = False
        while not valid_input:
            print(f"Possible moves pick a number between 0 and {nb_possible_moves - 1} : \n {possible_moves}")
            try:
                chosen_move = int(input())
            except ValueError:
                print(f'The input has to be an integer. Please try again.')
            else:
                if chosen_move in range(nb_possible_moves):
                    valid_input = True
                else:
                    print(f'This move is not valid. Please try again.')
        return possible_moves[chosen_move]
        
class PlayerSearchTreeAI(Player) :

    def __init__(self, ID, depthMax) -> None:
        Player.__init__(self, ID)
        self.depthMax = depthMax
        self.alignment_score_table = dict()
        self.build_alignment_score_table()
        self.state_score_table = dict()
        #print(self.alignment_score_table)
        #quit()
    
    def build_alignment_score_table(self):
        win_size = GameState.WIN_SIZE
        for i in range(win_size+1):
            if win_size-i > 1:
                self.alignment_score_table[(win_size-i, i)] = (win_size-i)*math.exp(win_size-i)
        
    def compute_alignments_score(self, player, other_player, window, gameState: GameState.GameState, grid) -> int:
            return self.alignment_score_table.get((window.count(player), window.count(None)), 0) - \
                        self.alignment_score_table.get((window.count(other_player), window.count(None)), 0)
  
    def compute_grid_quality_score(self, gameState: GameState.GameState) -> int:
        player = self.ID
        other_player = 1-self.ID
        score = 0
        grid = gameState.Grid
        win_size = GameState.WIN_SIZE
        grid_size = GameState.SIZE
        dif_size = grid_size - win_size
        # By moving a window of size win_size, check the alignments in ...
        for j in range(dif_size+1):
            for x in range(grid_size):
                # ... the columns of the X layers, and in ...
                for y in range(grid_size):
                    col = [grid[x][y][i] for i in range(grid_size)]
                    window = col[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                # ... the rows of the X layers, and in ...
                for z in range(grid_size):
                    row = [grid[x][i][z] for i in range(grid_size)]
                    window = row[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                # ... the diagonals of the X layers ...
                diag = [grid[x][i][i] for i in range(grid_size)]
                window = diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                diag = [grid[x][i][grid_size-1-i] for i in range(grid_size)]
                window = diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                for k in range(1,dif_size+1):
                    small_diag = [grid[x][i+k][i] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[x][i][i+k] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[x][i+k][grid_size-1-i] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[x][i][grid_size-1-i-k] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            for y in range(grid_size):
                for z in range(grid_size):
                    # ... the rows of the Z layers, and in ...
                    row = [grid[i][y][z] for i in range(grid_size)]
                    window = row[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                # ... the diagonals of the Z layers, and in ...
                diag = [grid[i][y][i] for i in range(grid_size)]
                window = diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                diag =  [grid[i][y][grid_size-1-i] for i in range(grid_size)]
                window = diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                for k in range(1,dif_size+1):
                    small_diag = [grid[i][y][i+k] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[i+k][y][i] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[i][y][grid_size-1-i-k] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[i+k][y][grid_size-1-i] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            for z in range(grid_size):
                # ... the diagonals of the Y layers, and in ...
                diag = [grid[i][i][z] for i in range(grid_size)]
                window = diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                diag = [grid[i][grid_size-1-i][z] for i in range(grid_size)]
                window = diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                for k in range(1,dif_size+1):
                    small_diag = [grid[i+k][i][z] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[i][i+k][z] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[i+k][grid_size-1-i][z] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                    small_diag = [grid[i][grid_size-1-i-k][z] for i in range(grid_size-k)]
                    window = small_diag[j:j+win_size]
                    score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            # ... the diagonals that cross the X, the Y and the Z layers
            diag = [grid[i][i][i] for i in range(grid_size)]
            window = diag[j:j+win_size]
            score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            diag = [grid[i][i][grid_size-1-i] for i in range(grid_size)]
            window = diag[j:j+win_size]
            score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            diag = [grid[grid_size-1-i][i][grid_size-1-i] for i in range(grid_size)]
            window = diag[j:j+win_size]
            score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            diag = [grid[grid_size-1-i][i][i] for i in range(grid_size)]
            window = diag[j:j+win_size]
            score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            for k in range(1,dif_size+1):
                small_diag = [grid[i][i+k][i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i][grid_size-1-i][grid_size-1-i-k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i][i][i+k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i][grid_size-1-i-k][grid_size-1-i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i+k][i+k][i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i+k][grid_size-1-i][grid_size-1-i-k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i+k][i][i+k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i+k][grid_size-1-i-k][grid_size-1-i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i][i][grid_size-1-i-k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[grid_size-1-i-k][grid_size-1-i-k][i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i][i+k][grid_size-1-i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[grid_size-1-i-k][grid_size-1-i][i+k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i+k][i][grid_size-1-i-k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[grid_size-1-i][grid_size-1-i-k][i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[i+k][i+k][grid_size-1-i] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
                small_diag = [grid[grid_size-1-i][grid_size-1-i][i+k] for i in range(grid_size-k)]
                window = small_diag[j:j+win_size]
                score += self.compute_alignments_score(player, other_player, window, gameState, grid)
            # Return final score
            return int(score)
        
    def MinMaxAlphaBetaPruning(self, gameState: GameState.GameState, depth, alpha, beta, maximizingPlayer) -> int:
        # Terminating condition
        if depth == 0 or gameState.checkEnd():
            # Use a transposition table to save the already computed game state
            if str(gameState.Grid) in self.state_score_table:
                return self.state_score_table[str(gameState.Grid)]
            else:
                score = self.compute_grid_quality_score(gameState)
                self.state_score_table[str(gameState.Grid)] = score
            return score
        # Maximizing player block
        if maximizingPlayer:
            bestValue = -float('inf')
            # Recur on all children
            for move in gameState.getPossibleMoves():
                new_gameState = gameState.copy()
                new_gameState.playLegalMove(move)
                value = self.MinMaxAlphaBetaPruning(new_gameState, depth - 1, alpha, beta, False)
                bestValue = max(bestValue, value)
                alpha = max(alpha, bestValue)
                # Alpha Beta Pruning
                if beta <= alpha:
                    break
            return bestValue
        # Minimizing player block
        else:
            bestValue = float('inf')
            # Recur on all children
            for move in gameState.getPossibleMoves():
                new_gameState = gameState.copy()
                new_gameState.playLegalMove(move)
                value = self.MinMaxAlphaBetaPruning(new_gameState, depth - 1, alpha, beta, True)
                bestValue = min(bestValue, value)
                beta = min(beta, bestValue)
                # Alpha Beta Pruning
                if beta <= alpha:
                    break
            return bestValue
        
    def random_max_index(self, values):
        max_index = []
        max_value = max(values)
        for i, value in enumerate(values):
            if value == max_value:
                max_index.append(i)
        return random.choice(max_index)

    def strategy(self, gameState: GameState.GameState) -> tuple:
        values = list()
        for move in gameState.getPossibleMoves():
            new_gameState = gameState.copy()
            new_gameState.playLegalMove(move)
            value = self.MinMaxAlphaBetaPruning(new_gameState, self.depthMax, -float("inf"), float("inf"), False)
             #print(move, value)
            values.append(value)
        # Choose a random state among those with the highest values
        random_max_index = self.random_max_index(values)
        bestMove = gameState.getPossibleMoves()[random_max_index]
        return bestMove
