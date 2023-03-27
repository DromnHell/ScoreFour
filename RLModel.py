
import GameState
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

class ScoreFourNN(nn.Module):
    def __init__(self):
        super(ScoreFourNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(GameState.SIZE**3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, GameState.SIZE**2)
        )
    def forward(self, x, player_id):
        #x = torch.cat([x, player_id.view(-1, 1)], dim = 1)
        x = x.view(-1, GameState.SIZE**3) 
        player_id = player_id.view(-1, 1)
        x = torch.cat([x, player_id], dim = 1)
        return self.layers(x)
    
def save_weights(model, file_path):
    '''
    Save the model
    '''
    torch.save(model.state_dict(), file_path)

def load_weights(model, file_path):
    '''
    Load the weights of the model
    '''
    model.load_state_dict(torch.load(file_path))
    # Put the template in evaluation mode to disable training-specific features
    model.eval()

def plot_metrics(rewards, num_moves, window_size):
    '''
    Plot somes metrics
    '''

    rewards_smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode = 'valid')
    num_moves_smoothed = np.convolve(num_moves, np.ones(window_size) / window_size, mode = 'valid')

    fig, ax1 = plt.subplots()

    ax1.plot(rewards_smoothed, color = 'blue', label = 'Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc = 'upper left')

    ax2 = ax1.twinx()
    ax2.plot(num_moves_smoothed, color = 'red', label = 'Number of Moves')
    ax2.set_ylabel('Number of Moves')
    ax2.legend(loc = 'upper right')

    plt.show()

def grid_to_input(grid):
    '''
    Convert the grid in Numpy array
    '''
    input_tensor = np.zeros((GameState.SIZE**3,))
    for i, layer in enumerate(grid):
        for j, col in enumerate(layer):
            for k, cell in enumerate(col):
                if cell is not None:
                    input_tensor[i * GameState.SIZE**2 + j * GameState.SIZE + k] = cell + 1
    return torch.tensor(input_tensor.flatten(), dtype = torch.float).unsqueeze(0)

def select_action(current_player_id, gameState: GameState.GameState, model, epsilon):
    '''
    Select the action
    '''
    if random.random() < epsilon:
        possible_moves = gameState.getPossibleMoves()
        move_index = random.choice(range(len(possible_moves)))
        return move_index, gameState.getPossibleMoves()[move_index]
    else:
        with torch.no_grad():
            state_input = grid_to_input(gameState.Grid)
            player_input = torch.tensor([current_player_id], dtype = torch.float).unsqueeze(0)
            q_values = model(state_input, player_input)
            possible_move_index = [move[1] for move in gameState.getPossibleMoves()]
            possible_q_values = q_values[0][possible_move_index]
            best_move_index = int(torch.argmax(possible_q_values))
            return best_move_index, gameState.getPossibleMoves()[best_move_index]

def update_model(model, optimizer, loss_fn, states, players, actions, rewards, next_states, next_players, dones, gamma = 0.99):
    '''
    Update the model
    '''
    q_values = model(states, players)
    played_q_values = q_values.gather(1, actions.view(-1, 1)).squeeze()

    next_q_values = model(next_states, next_players)
    max_next_q_values, _ = torch.max(next_q_values, dim = 1)
    target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    played_q_values = played_q_values.view(-1, 1)
    target_q_values = target_q_values.view(-1, 1)

    loss = loss_fn(played_q_values, target_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_self_play(model, episodes, epsilon = 1, learning_rate = 0.001):
    '''
    Train the model with self-play
    '''
    rewards = []
    num_moves = []

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_fn = nn.SmoothL1Loss()

    for episode in range(episodes + 1):
        print(episode)

        episode_reward = 0
        episode_moves = 0

        state = GameState.GameState()
        current_player_id = 0
        done = False

        while not done:

            action_index, action = select_action(current_player_id, state, model, epsilon)

            next_state = state.copy()
            next_state.playLegalMove(action[0])

            if next_state.getWinner() is not None:
                if current_player_id == 0:
                    reward = 1
                else:
                    reward = -1
            else:
                reward = 0

            done = True if next_state.checkEnd() == True else False

            next_player_id = 1 - current_player_id

            state_input = grid_to_input(state.Grid)
            player_input = torch.tensor([current_player_id], dtype = torch.float).unsqueeze(0)
            action_tensor = torch.tensor([action_index], dtype = torch.int64)
            reward_tensor = torch.tensor([reward], dtype = torch.float32)
            next_state_input = grid_to_input(next_state.Grid)
            next_player_input = torch.tensor([next_player_id], dtype = torch.float).unsqueeze(0)
            done_tensor = torch.tensor([done], dtype = torch.float32)

            update_model(model, optimizer, loss_fn, state_input, player_input, action_tensor, reward_tensor, next_state_input, next_player_input, done_tensor)

            state = next_state

            current_player_id = next_player_id

            episode_reward = reward
            episode_moves += 1

            if done:
                break

        if epsilon > 0.05:
            epsilon *= 0.995

        rewards.append(episode_reward)
        num_moves.append(episode_moves)

        if episode % 1000 == 0:
            print(f'Episode {episode}: reward = {episode_reward}, moves = {episode_moves}')

    return rewards, num_moves

if __name__ == "__main__":    

    model = ScoreFourNN()

    rewards, num_moves = train_self_play(model, episodes = 1000000)

    plot_metrics(rewards, num_moves, 10000)

    file_path = "score_four_RL_weights.pth"
    save_weights(model, file_path)



