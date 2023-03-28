'''
This script is part of the Score Four program. It contains all the classes of the different possible players :
- A random player AI,
- A human player,
- A tree search tree AI player,
- A deep reinforcement learning AI player.
'''

from abc import ABC, abstractmethod
import GameState
import RLbasic
import random
import math

current_IDs = list()


class Player(ABC):
    
    def __init__(self, ID) -> None:
        if ID not in (0, 1, "helper"):
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

    @abstractmethod
    def receive_last_feedback(self, old_gameState, old_player, move, reward, gameState, player):
        pass


class PlayerRandom(Player):

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
        self.name = "RNDAI"
    
    def strategy(self, gameState: GameState.GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())[0]
    
    def receive_last_feedback(self, winning_gameState: GameState.GameState, winner, last_move, reward, gameState: GameState.GameState, looser):
        pass
    

class PlayerHuman(Player) : 

    def __init__(self, ID) -> None:
        Player.__init__(self, ID)
        self.name = "HUMAN"

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
        return possible_moves[chosen_move][0]
    
    def receive_last_feedback(self, winning_gameState: GameState.GameState, winner, last_move, reward, gameState: GameState.GameState, looser):
        pass
        

class PlayerSearchTreeAI(Player) :

    def __init__(self, ID, depthMax = 0, epsilon = None) -> None:
        Player.__init__(self, ID)
        self.name = f"STAI{depthMax}"
        self.depthMax = depthMax
        self.epsilon = epsilon
        self.alignment_score_table = dict()
        self.build_alignment_score_table()
        self.state_score_table = dict()
    
    def build_alignment_score_table(self):
        '''
        Build the alignment score table.
        The higher the ratio "pawn /empty square" of the window, the higher the score of the window.
        The distribution of scores follows an exponential function.
        '''
        win_size = GameState.WIN_SIZE
        for i in range(win_size+1):
            if win_size-i > 0:
                self.alignment_score_table[(win_size-i, i)] = int((win_size-i)*math.exp(win_size-i))
        
    def compute_alignments_score(self, player, other_player, window) -> int:
        '''
        Compute the alignements score windows by subtracting the score of player 0 (or 1) from that of player 1 (or 0 respectively).
        '''
        return self.alignment_score_table.get((window.count(player), window.count(None)), 0) - \
                    self.alignment_score_table.get((window.count(other_player), window.count(None)), 0)
    
    def compute_last_move_score(self, gameState: GameState.GameState) -> int:
        '''
        Calculates the score of the last move by calculating the score of the 13 segments that intersect in its coordinate,
        and in the top coordinate.
        '''
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
        '''
        Min max algorithm with alpha beta pruning.
        '''
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
        '''
        Choose randomly the index betwen the max identical ones.
        '''
        max_index = []
        max_value = max(values)
        for i, value in enumerate(values):
            if value == max_value:
                max_index.append(i)
        return random.choice(max_index)

    def strategy(self, gameState: GameState.GameState) -> tuple:
        '''
        Evaluates each movement recursively according to a given depth.
        '''
        # Allow the AI to play randomly. Only usefull to help RL training
        if self.epsilon != None and random.random() < self.epsilon:
            return random.choice(gameState.getPossibleMoves())[0]
        else:
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
    
    def receive_last_feedback(self, winning_gameState: GameState.GameState, winner, last_move, reward, gameState: GameState.GameState, looser):
        pass


class PlayerRLAI(Player) :

    def __init__(self, ID, learn = False, weights_file = None) -> None:
        Player.__init__(self, ID)
        self.name = "RLAI"
        self.learn = learn
        self.weights_file = weights_file
        self.model = RLbasic.ScoreFourNN()
        self.optimizer = RLbasic.optim.Adam(self.model.parameters(), lr = 0.001)
        self.loss_fn = RLbasic.nn.SmoothL1Loss()
        self.done = False
        self.epsilon = 1
        if learn == False:
            self.epsilon = 0
        if weights_file != None:
            RLbasic.load_weights(self.model, self.weights_file)

    def strategy(self, gameState: GameState.GameState) -> tuple:
        '''
        Two modes : 
        1) play to learn using Deep RL,
        2) play to win using previous trained Deep RL model.
        '''
        player = 0 if gameState.IsPlayerZeroTurn == True else 1
        if self.learn == True:
            move_index, bestMove = RLbasic.select_action(0, player, gameState, self.model, self.epsilon)
            new_gameState = gameState.copy()
            new_gameState.playLegalMove(bestMove[0])
            reward = 1 if new_gameState.getWinner() is not None else 0
            done = True if new_gameState.checkEnd() == True else False
            next_player = 1 - player
            state_input = RLbasic.grid_to_input(gameState.Grid)
            player_input = RLbasic.torch.tensor([player], dtype = RLbasic.torch.float).unsqueeze(0)
            action_tensor = RLbasic.torch.tensor([move_index], dtype = RLbasic.torch.int64)
            reward_tensor = RLbasic.torch.tensor([reward], dtype = RLbasic.torch.float32)
            next_state_input = RLbasic.grid_to_input(new_gameState.Grid)
            next_player_input = RLbasic.torch.tensor([next_player], dtype = RLbasic.torch.float).unsqueeze(0)
            done_tensor = RLbasic.torch.tensor([done], dtype = RLbasic.torch.float32)
            RLbasic.update_model(0, self.model, self.optimizer, self.loss_fn, state_input, player_input, action_tensor, reward_tensor, next_state_input, next_player_input, done_tensor)
        else:
            move_index, bestMove = RLbasic.select_action(0, player, gameState, self.model, epsilon = 0)
        return bestMove[0]
    
    def receive_last_feedback(self, old_gameState: GameState.GameState, winner, last_move, reward, winning_gameState: GameState.GameState, looser):
        '''
        When the opponent wins and the episode is about to end, the model learns one
        last time to take into account the move that led him to the defeat.
        '''
        if self.learn == True:
            for i in range(len(old_gameState.getPossibleMoves())):
                if old_gameState.getPossibleMoves()[i][0] == last_move:
                    index_last_move = i
            old_state_input = RLbasic.grid_to_input(old_gameState.Grid)
            winner_input = RLbasic.torch.tensor([winner], dtype = RLbasic.torch.float).unsqueeze(0)
            action_tensor = RLbasic.torch.tensor([index_last_move], dtype = RLbasic.torch.int64)
            reward_tensor = RLbasic.torch.tensor([reward], dtype = RLbasic.torch.float32)
            winning_state_input = RLbasic.grid_to_input(winning_gameState.Grid)
            looser_input = RLbasic.torch.tensor([looser], dtype = RLbasic.torch.float).unsqueeze(0)
            done_tensor = RLbasic.torch.tensor([1], dtype = RLbasic.torch.float32)
            RLbasic.update_model(0, self.model, self.optimizer, self.loss_fn, old_state_input, winner_input, action_tensor, reward_tensor, winning_state_input, looser_input, done_tensor)