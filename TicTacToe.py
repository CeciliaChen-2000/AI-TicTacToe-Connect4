import math
import pickle
import random
import time
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt


class TicTacToe:
    def __init__(self):
        self.board = self.init_board()
        self.winner = None
        self.move_history = []

    @staticmethod
    def init_board():
        return [[' ' for _ in range(3)] for _ in range(3)]

    def print_board(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                print("|", self.board[i][j], end=" ")
            print("|")
        print('')
        time.sleep(0.1)

    def make_move(self, position, player):
        # position = [x, y]
        x = position[0]
        y = position[1]
        if self.board[x][y] == ' ':
            state_key = tuple(tuple(row) for row in self.board)
            self.board[x][y] = player
            new_state_key = tuple(tuple(row) for row in self.board)
            self.move_history.append((state_key, position, new_state_key))
            if self.is_winner(position, player):
                self.winner = player
            return True
        return False

    def is_winner(self, position, player):
        # position = (x, y)
        x = position[0]
        y = position[1]
        winning_condition = [player, player, player]
        if self.board[x] == winning_condition:  # check row
            return True
        if [self.board[i][y] for i in range(3)] == winning_condition:  # check column
            return True
        if x == y and [self.board[i][i] for i in range(3)] == winning_condition:  # check diagonal
            return True
        if x + y == 2 and [self.board[i][2 - i] for i in range(3)] == winning_condition:  # check the other diagonal
            return True
        return False

    @staticmethod
    def is_gonna_win(board, position, player):
        # position = (x, y)
        x = position[0]
        y = position[1]
        winning_condition = [player, player, player]
        if board[x] == winning_condition:  # check row
            return True
        if [board[i][y] for i in range(3)] == winning_condition:  # check column
            return True
        if x == y and [board[i][i] for i in range(3)] == winning_condition:  # check diagonal
            return True
        if x + y == 2 and [board[i][2 - i] for i in range(3)] == winning_condition:  # check the other diagonal
            return True
        return False

    def is_available_move(self):
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == ' ':
                    return True
        return False

    def num_available_moves(self):
        available_moves = []
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == ' ':
                    available_moves.append((x, y))
        return len(available_moves)

    def get_available_moves(self):
        available_moves = []
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == ' ':
                    available_moves.append((x, y))
        return available_moves


class Player:
    def __init__(self, player):
        self.player = player

    def get_move(self, game):
        pass


class HumanPlayer(Player):
    def __init__(self, player):
        super().__init__(player)

    def get_move(self, game):
        valid_position = False
        position = None
        while not valid_position:
            try:
                x, y = input(f"{self.player}'s turn. Input position (row,column): ").split(',')
                position = (int(x), int(y))
                if position not in game.get_available_moves():
                    raise ValueError
                valid_position = True
            except ValueError:
                print('Invalid position. Try again.')
        return position


class DefaultComputerPlayer(Player):
    def __init__(self, player):
        super().__init__(player)

    def get_move(self, game):
        if game.num_available_moves() == 9:  # random choose first step
            position = random.choice(game.get_available_moves())
        else:
            position = self.default(game)['position']
        return position

    def default(self, game):
        self_player = self.player  # yourself
        opponent_player = 'O' if self.player == 'X' else 'X'

        # first check if the previous move is a winner
        if game.winner == opponent_player:
            return {'position': None}
        elif not game.is_available_move():  # check if there is available move
            return {'position': None}

        for x in range(len(game.board)):
            for y in range(len(game.board[0])):
                if game.board[x][y] == ' ':
                    # Check if any move can immediately win the game
                    board_copy = deepcopy(game.board)
                    board_copy[x][y] = self_player
                    if game.is_gonna_win(board_copy, (x, y), self_player):
                        return {'position': (x, y)}

                    # Check if any move can immediately block the other player from winning
                    board_copy = deepcopy(game.board)
                    board_copy[x][y] = opponent_player
                    if game.is_gonna_win(board_copy, (x, y), opponent_player):
                        return {'position': (x, y)}

        # Random choose a blank position
        return {'position': random.choice(game.get_available_moves())}


class MinimaxComputerPlayer(Player):
    def __init__(self, player):
        super().__init__(player)

    def get_move(self, game):
        if game.num_available_moves == 9:
            position = random.choice(game.get_available_moves())
        else:
            position = self.minimax(game, self.player)['position']
        return position

    # def minimax(self, state, player):
    def minimax(self, state, player, alpha=-math.inf, beta=math.inf):
        max_player = self.player  # yourself
        opponent_player = 'O' if player == 'X' else 'X'

        # first check if the previous move is a winner
        if state.winner == opponent_player:
            return {'position': None,
                    'score': 1 * (state.num_available_moves() + 1) if opponent_player == max_player else -1 * (
                            state.num_available_moves() + 1)}
        elif not state.is_available_move():  # 没有空格
            return {'position': None, 'score': 0}

        if player == max_player:  # 当前玩家，
            best = {'position': None, 'score': -math.inf}  # each score should maximize
        else:  # 对手
            best = {'position': None, 'score': math.inf}  # each score should minimize
        for possible_move in state.get_available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, opponent_player)  # simulate a game after making that move

            # undo move
            state.board[possible_move[0]][possible_move[1]] = ' '
            state.winner = None
            sim_score['position'] = possible_move  # this represents the move optimal next move

            if player == max_player:  # X is max player
                if sim_score['score'] > best['score']:
                    best = sim_score
                alpha = max(alpha, best['score'])
                if beta <= alpha:
                    break
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
                beta = min(beta, best['score'])
                if beta <= alpha:
                    break
        return best


class QlearningComputerPlayer(Player):
    def __init__(self, player, q_values):
        super().__init__(player)
        self.q_values = q_values

    def get_move(self, game):
        if game.num_available_moves == 9:
            position = random.choice(game.get_available_moves())
        else:
            position = self.qlearning(game)['position']
        return position

    def qlearning(self, state):
        state_key = tuple(map(tuple, state.board))
        if state_key not in self.q_values:
            return {'position': random.choice(state.get_available_moves())}
        else:
            q_values = self.q_values[state_key]
            q_values = {k: v for k, v in q_values.items() if k in state.get_available_moves()}
            if not q_values:
                action = random.choice(state.get_available_moves())
            else:
                action = max(q_values.items(), key=lambda x: x[1])[0]
            return {'position': action}


class QLearning(Player):
    def __init__(self, player, epsilon=0.5, alpha=0.1, gamma=0.9):
        super().__init__(player)
        self.q_values = {}  # 存储Q值的字典
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子

    def get_move(self, game):
        if game.num_available_moves == 9:
            position = random.choice(game.get_available_moves())
        else:
            position = self.qlearning(game)['position']
        return position

    def qlearning(self, state):
        # 初始状态
        state_key = tuple(map(tuple, state.board))
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        # 选择动作
        if np.random.uniform() < self.epsilon:
            possible_moves = state.get_available_moves()
            action = random.choice(possible_moves)
        else:
            q_values = self.q_values[state_key]
            q_values = {k: v for k, v in q_values.items() if k in state.get_available_moves()}
            if not q_values:
                action = random.choice(state.get_available_moves())
            else:
                action = max(q_values.items(), key=lambda x: x[1])[0]
        return {'position': action}

    def get_q_value(self, state, action):
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = 0.0
        return self.q_values[state][action]

    def update_q_table(self, game):
        winner = game.winner
        if winner == self.player:
            reward = 1
        elif winner is None:
            reward = 0
        else:
            reward = -1

        for state_key, action, new_state_key in game.move_history[::-1]:
            if state_key not in self.q_values:
                self.q_values[state_key] = {}
            if new_state_key not in self.q_values:
                self.q_values[new_state_key] = {}

            max_q_value = max(self.q_values[new_state_key].values()) if self.q_values[new_state_key] else 0.0
            old_q_value = self.get_q_value(state_key, action)
            new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_q_value)
            # new_q_value = max(-1.0, min(1.0, new_q_value))
            self.q_values[state_key][action] = new_q_value

            # reward = new_q_value
            reward = new_q_value if reward is None else self.get_q_value(state_key, action)

    # def save_q_table(self, file_path):
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(self.q_values, f)


def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board()

    current_player = 'X'
    while game.is_available_move():
        if current_player == 'O':
            position = o_player.get_move(game)
        else:
            position = x_player.get_move(game)
        if game.make_move(position, current_player):
            if print_game:
                print(current_player + ' makes a move to position {}'.format(position))
                game.print_board()

            if game.winner:
                if print_game:
                    print(current_player + ' wins!')
                return current_player  # ends the loop and exits the game
            current_player = 'O' if current_player == 'X' else 'X'  # switches player

    if print_game:
        print('It\'s a tie!')
        return None


def train_qlearning_tictactoe(epsilon, alpha, gamma, num_epochs=100000):
    x_player = QLearning('X', epsilon, alpha, gamma)
    o_player = QLearning('O', epsilon, alpha, gamma)
    for i in range(num_epochs):
        game = TicTacToe()
        # players = ['X', 'O']
        current_player = 'X'
        while game.is_available_move():
            # get the next move for current player
            if current_player == 'O':
                position = o_player.get_move(game)
            else:
                position = x_player.get_move(game)

            game.make_move(position, current_player)  # make the move
            current_player = 'O' if current_player == 'X' else 'X'  # switches player
        x_player.update_q_table(game)
        o_player.update_q_table(game)

        if (i + 1) % (num_epochs // 100) == 0:
            print(f"Training progress: {i + 1}/{num_epochs}")

    print('num_epochs', num_epochs)
    return x_player
    # x_player.save_q_table('q_table_tictactoe.pkl')


def evaluate(rounds):
    num_rounds = rounds

    # Minimax vs Default Opponent
    minimax_vs_default_results = []
    minimax_vs_default_win = []
    minimax_vs_default_lose = []
    start_time = time.time()
    for n in range(num_rounds):
        player1 = MinimaxComputerPlayer('X')
        player2 = DefaultComputerPlayer('O')
        minimax_vs_default_result = play(TicTacToe(), player1, player2, False)
        minimax_vs_default_results.append(minimax_vs_default_result)
        minimax_vs_default_win.append(minimax_vs_default_results.count(player1.player) / (n+1))
        minimax_vs_default_lose.append(minimax_vs_default_results.count(player2.player) / (n+1))
    end_time = time.time()
    total_time = end_time - start_time
    plt.plot(minimax_vs_default_win, label='Rate (Minimax Win)')
    plt.plot(minimax_vs_default_lose, label='Rate (Default Opponent Win)')
    plt.legend()
    plt.title(f'Minimax vs Default Opponent, sec / round: {total_time/num_rounds:.5f}')
    plt.show()

    # Q-Learning vs Default Opponent
    with open('q_table_tictactoe.pkl', 'rb') as f:
        q_values_table = pickle.load(f)
    qlearning_vs_default_results = []
    qlearning_vs_default_win = []
    qlearning_vs_default_lose = []
    start_time = time.time()
    for n in range(num_rounds):
        player1 = QlearningComputerPlayer('X', q_values_table)
        player2 = DefaultComputerPlayer('O')
        qlearning_vs_default_result = play(TicTacToe(), player1, player2, False)
        qlearning_vs_default_results.append(qlearning_vs_default_result)
        qlearning_vs_default_win.append(qlearning_vs_default_results.count(player1.player) / (n + 1))
        qlearning_vs_default_lose.append(qlearning_vs_default_results.count(player2.player) / (n + 1))
    end_time = time.time()
    total_time = end_time - start_time
    plt.plot(qlearning_vs_default_win, label='Rate (Q-Learning Win)')
    plt.plot(qlearning_vs_default_lose, label='Rate (Default Opponent Win)')
    plt.legend()
    plt.title(f'Q-Learning vs Default Opponent, sec / round: {total_time/num_rounds:.5f}')
    plt.show()

    # Minimax vs Q-Learning
    minimax_vs_qlearning_results = []
    minimax_vs_qlearning_win = []
    minimax_vs_qlearning_lose = []
    start_time = time.time()
    for n in range(num_rounds):
        player1 = MinimaxComputerPlayer('X')
        player2 = QlearningComputerPlayer('O', q_values_table)
        minimax_vs_qlearning_result = play(TicTacToe(), player1, player2, False)
        minimax_vs_qlearning_results.append(minimax_vs_qlearning_result)
        minimax_vs_qlearning_win.append(minimax_vs_qlearning_results.count(player1.player) / (n + 1))
        minimax_vs_qlearning_lose.append(minimax_vs_qlearning_results.count(player2.player) / (n + 1))
    end_time = time.time()
    total_time = end_time - start_time
    plt.plot(minimax_vs_qlearning_win, label='Rate (Minimax Win)')
    plt.plot(minimax_vs_qlearning_lose, label='Rate (Q-Learning Win)')
    plt.legend()
    plt.title(f'Minimax vs Q-Learning, sec / round: {total_time/num_rounds:.5f}')
    plt.show()


def execute():
    with open('q_table_tictactoe.pkl', 'rb') as f:
        q_values_table = pickle.load(f)

    player1 = None
    player2 = None

    while player1 is None:
        try:
            input_player1 = int(input('Choose role for X player (1-Human, 2-DefaultOpponent, 3-Minimax, 4-QLearning):'))
            if input_player1 == 1:
                player1 = HumanPlayer('X')
            elif input_player1 == 2:
                player1 = DefaultComputerPlayer('X')
            elif input_player1 == 3:
                player1 = MinimaxComputerPlayer('X')
            elif input_player1 == 4:
                player1 = QlearningComputerPlayer('X', q_values_table)
            else:
                raise ValueError
        except ValueError:
            print('Please enter number from 1-4.')

    while player2 is None:
        try:
            input_player2 = int(input('Choose role for O player (1-Human, 2-DefaultOpponent, 3-Minimax, 4-QLearning):'))
            if input_player2 == 1:
                player2 = HumanPlayer('O')
            elif input_player2 == 2:
                player2 = DefaultComputerPlayer('O')
            elif input_player2 == 3:
                player2 = MinimaxComputerPlayer('O')
            elif input_player2 == 4:
                player2 = QlearningComputerPlayer('O', q_values_table)
            else:
                raise ValueError
        except ValueError:
            print('Please enter number from 1-4.')

    play(TicTacToe(), player1, player2, True)


if __name__ == '__main__':
    execute()
    evaluate(int(input('Enter the number of rounds to evaluate algorithms:')))

