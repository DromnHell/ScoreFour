from abc import ABC, abstractmethod
import GameState
import RLModel
import random
import math

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

class PlayerFake(Player):

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
    
    def strategy(self, gameState: GameState.GameState) -> tuple:
        pass

class PlayerRandom(Player):

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
    
    def strategy(self, gameState: GameState.GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())[0]
    
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
        return possible_moves[chosen_move[0]]
        
class PlayerSearchTreeAI(Player) :

    def __init__(self, ID, depthMax) -> None:
        Player.__init__(self, ID)
        self.depthMax = depthMax
        self.alignment_score_table = dict()
        self.build_alignment_score_table()
        self.state_score_table = dict()
    
    def build_alignment_score_table(self):
        win_size = GameState.WIN_SIZE
        for i in range(win_size+1):
            if win_size-i > 0:
                self.alignment_score_table[(win_size-i, i)] = int((win_size-i)*math.exp(win_size-i))
        
    def compute_alignments_score(self, player, other_player, window) -> int:
            return self.alignment_score_table.get((window.count(player), window.count(None)), 0) - \
                        self.alignment_score_table.get((window.count(other_player), window.count(None)), 0)
    
    def compute_last_move_score(self, gameState: GameState.GameState) -> int:
        player = self.ID
        other_player = 1-self.ID
        lastMove = gameState.LastMove 
        grid = gameState.Grid
        win_size = GameState.WIN_SIZE
        grid_size = GameState.SIZE
        dif_size = grid_size - win_size
        score = 0
        # If it's the first move, return a null score
        if lastMove is None:
            return 0
        else:
            x, y, z = lastMove
        # From the coordinate of the last movement and its z+1 coordinate, record the elements of ...
        list_z = [z] if z+1 == grid_size else [z, z+1]
        for z in list_z:
            segments = []
            segments.extend([
                # ... the X layer column, and of ...
                [grid[x][y][j] for j in range(grid_size)],
                # ... the X layer row, and of ...
                [grid[x][j][z] for j in range(grid_size)],
                # ... the 2 X layer diagonals, and of ...
                [grid[x][i][j] for i in range(y - 1, -1, -1) for j in range(z - 1, -1, -1) if i - y == j - z][::-1] + \
                    [grid[x][i][j] for i in range(y, grid_size) for j in range(z, grid_size) if i - y == j - z],
                [grid[x][i][j] for i in range(y, -1, -1) for j in range(z, grid_size) if i - y == z - j][::-1] + \
                    [grid[x][i][j] for i in range(y, grid_size) for j in range(z - 1, -1, -1) if i - y == z - j],
                # ... the Y layer row, and of ...
                [grid[j][y][z] for j in range(grid_size)],
                # ... the 2 Y layer diagonals, and of ...
                [grid[i][y][j] for i in range(x - 1, -1, -1) for j in range(z - 1, -1, -1) if i - x == j - z][::-1] + \
                    [grid[i][y][j] for i in range(x, grid_size) for j in range(z, grid_size) if i - x == j - z],
                [grid[i][y][j] for i in range(x, -1, -1) for j in range(z, grid_size) if i - x == z - j][::-1] + \
                    [grid[i][y][j] for i in range(x, grid_size) for j in range(z - 1, -1, -1) if i - x == z - j],
                # ... the 2 Z layer diagonals, and of ...
                [grid[i][j][z] for i in range(x - 1, -1, -1) for j in range(y - 1, -1, -1) if i - x == j - y][::-1] + \
                    [grid[i][j][z] for i in range(x, grid_size) for j in range(y, grid_size) if i - x == j - y],
                [grid[i][j][z] for i in range(x, -1, -1) for j in range(y, grid_size) if i - x == y - j][::-1] + \
                    [grid[i][j][z] for i in range(x, grid_size) for j in range(y - 1, -1, -1) if i - x == y - j],
                # ... the 4 diagonals that cross the X, the Y and the Z layers.
                [grid[i][j][k] for i in range(x - 1, -1, -1) for j in range(y - 1, -1, -1) for k in range(z - 1, -1, -1) if i - x == j - y == k - z][::-1] + \
                    [grid[i][j][k] for i in range(x, grid_size) for j in range(y, grid_size) for k in range(z, grid_size) if i - x == j - y == k - z],
                [grid[i][j][k] for i in range(x - 1, -1, -1) for j in range(y - 1, -1, -1) for k in range(z, grid_size)if x - i == y - j == k - z][::-1] + \
                    [grid[i][j][k] for i in range(x, grid_size) for j in range(y, grid_size) for k in range(z, -1, -1)if i - x == j - y == z - k],
                [grid[i][j][k] for i in range(x - 1, -1, -1)for j in range(y, grid_size) for k in range(z - 1, -1, -1)if x - i == j - y == z - k][::-1] + \
                    [grid[i][j][k] for i in range(x, grid_size) for j in range(y, -1, -1) for k in range(z, grid_size)if i - x == y - j == k - z],
                [grid[i][j][k] for i in range(x - 1, -1, -1)for j in range(y, grid_size) for k in range(z, grid_size)if x - i == j - y == k - z][::-1] + \
                    [grid[i][j][k] for i in range(x, grid_size) for j in range(y, -1, -1)for k in range(z, -1, -1)if i - x == y - j == z - k],
                ])
            # By moving a window of size "win_size" in those 13 segments, compute their alignments scores.
            for segment in segments:
                for i in range(dif_size + 1):
                    window = segment[i:i + win_size]
                    score += self.compute_alignments_score(player, other_player, window)
        # Return the final score
        return score
    
    def MinMaxAlphaBetaPruning(self, gameState: GameState.GameState, depth, alpha, beta, maximizingPlayer) -> int:
        # Terminating condition
        if depth == 0 or gameState.checkEnd():
            # Use a transposition table to save the already computed game state
            if str(gameState.Grid) in self.state_score_table:
                return self.state_score_table[str(gameState.Grid)]
            else:
                score = self.compute_last_move_score(gameState)
                self.state_score_table[str(gameState.Grid)] = score
            return score
        # Maximizing player block
        if maximizingPlayer:
            bestValue = -float('inf')
            # Recur on all children
            for move in gameState.getPossibleMoves():
                new_gameState = gameState.copy()
                new_gameState.playLegalMove(move[0])
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
                new_gameState.playLegalMove(move[0])
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
        values = []
        # For each possibles moves,  compute its value
        for move in gameState.getPossibleMoves():
            new_gameState = gameState.copy()
            new_gameState.playLegalMove(move[0])
            value = self.MinMaxAlphaBetaPruning(new_gameState, self.depthMax, -float("inf"), float("inf"), False)
            #print(move, value)
            values.append(value)
        # Choose a random state among those with the highest values
        random_max_index = self.random_max_index(values)
        bestMove = gameState.getPossibleMoves()[random_max_index][0]
        return bestMove
    

class PlayerRLAI(Player) :

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
        # Load the weights of the trained model
        self.model = RLModel.ScoreFourNN()
        RLModel.load_weights(self.model, "score_four_RL_weights.pth")

    def strategy(self, gameState: GameState.GameState) -> tuple:
        player = 0 if gameState.IsPlayerZeroTurn == True else 1
        bestMove = RLModel.select_action(player, gameState, self.model, epsilon = 0)[1][0]
        return bestMove