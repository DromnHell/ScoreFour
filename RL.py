import numpy as np
import tensorflow as tf

# Grille de jeu
game_board = np.array([
    [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]],
    [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]],
    [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]],
    [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
])

# Fonction d'état
def get_state(board):
    state = tuple(tuple(tuple(x) for x in row) for row in board)
    return state

# Fonction d'action
def get_valid_actions(board):
    valid_actions = []
    for col in range(4):
        if board[0, col, 0] is None:
            valid_actions.append(col)
    return valid_actions

# Fonction de récompense
def get_reward(board, player):
    # Vérifie s'il y a un gagnant
    winner = check_winner(board)
    if winner is not None:
        if winner == player:
            return 1.0
        else:
            return -1.0
    # Vérifie s'il y a un match nul
    elif check_draw(board):
        return 0.0
    else:
        return None

# Réseau de neurones
class ScoreFourNetwork(tf.keras.Model):
    def __init__(self):
        super(ScoreFourNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation='tanh')
        self.policy = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        value = self.value(x)
        policy = self.policy(x)
        return value, policy

# Initialisation du réseau
model = ScoreFourNetwork()

# Fonction de perte
def compute_loss(model, state, action, reward, next_state, terminal, gamma):
    state = np.array(state)[np.newaxis, ...]
    next_state = np.array(next_state)[np.newaxis, ...]

    with tf.GradientTape() as tape:
        value, policy = model(state)
        next_value, _ = model(next_state)

        # Calcul de la valeur cible
        target = reward + gamma * next_value * (1 - terminal)

        # Calcul de la perte de la valeur
        value_loss = tf.math.square(target - value)

        # Calcul de la perte de la politique
        log_prob = tf.math.log(policy + 1e-20)
        action_one_hot = tf.one_hot(action, depth=4)
        log_prob = tf.reduce_sum(log_prob * action_one_hot, axis=1, keepdims=True)
        advantage = target - value
        policy_loss = -log_prob * tf.stop_gradient(advantage)

        # Calcul de la perte totale
        total_loss = 0.5 * value_loss + policy_loss - 0.001 * tf.reduce_mean(log_prob)

    # Calcul du gradient
    grads = tape.gradient(total_loss, model.trainable_variables)

    return total_loss, grads

# Optimiseur
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Paramètres de l'algorithme d'apprentissage par renforcement profond
gamma = 0.99  # facteur d'actualisation
epsilon = 0.1  # probabilité d'exploration
num_episodes = 1000  # nombre d'épisodes

# Algorithme d'apprentissage par renforcement profond
for episode in range(num_episodes):
    # Initialisation de l'état de départ
    state = get_state(game_board)
    player = 0

    while True:
        # Choix de l'action
        valid_actions = get_valid_actions(game_board)
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(valid_actions)
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            _, policy = model(state_tensor)
            policy = np.array(policy)[0]
            policy_valid = policy[valid_actions]
            action = np.random.choice(len(policy_valid), p=policy_valid / np.sum(policy_valid))
            action = valid_actions[action]

        # Mise à jour de la grille de jeu
        for row in range(4):
            if game_board[row, action, 3] is None:
                game_board[row, action, 3] = player
                break

        # Récupération de la récompense
        reward = get_reward(game_board, player)
        if reward is not None:
            # Si la partie est terminée, met à jour le réseau de neurones et sort de la boucle
            terminal = 1
            _, next_policy = model(tf.convert_to_tensor(get_state(game_board), dtype=tf.float32))
            next_policy = np.array(next_policy)[0]
            next_policy_valid = next_policy[get_valid_actions(game_board)]
            total_loss, grads = compute_loss(model, state, action, reward, get_state(game_board), terminal, gamma)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            break
        else:
            terminal = 0

        # Mise à jour de l'état suivant
        next_state = get_state(game_board)

        # Choix de l'action suivante
        state_tensor = tf.convert_to_tensor(next_state, dtype=tf.float32)
        _, next_policy = model(state_tensor)
        next_policy = np.array(next_policy)[0]
        next_policy_valid = next_policy[get_valid_actions(game_board)]
        next_action = np.random.choice(len(next_policy_valid), p=next_policy_valid / np.sum(next_policy_valid))
        next_action = get_valid_actions(game_board)[next_action]

        # Mise à jour du réseau de neurones
        total_loss, grads = compute_loss(model, state, action, reward, next_state, terminal, gamma)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Passage à l
        state = next_state
        action = next_action
        player = 1 - player


