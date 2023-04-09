import math
import pickle
import random
import time
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt


class Connect4:
    def __init__(self):
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.board = self.create_board()
        self.winner = None
        self.move_history = []

    def create_board(self):
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))

    def drop_piece(self, column, piece):
        if self.is_valid_location(column):
            state_key = tuple(tuple(row) for row in self.board)

            row = self.get_next_open_row(column)
            self.board[row][column] = piece

            new_state_key = tuple(tuple(row) for row in self.board)
            self.move_history.append((state_key, column, new_state_key))

            if self.is_winner(piece):
                self.winner = piece
            return True
        return False

    def is_valid_location(self, column):
        if column is None:
            return False
        if 0 <= column <= self.COLUMN_COUNT - 1:
            return self.board[self.ROW_COUNT - 1][column] == 0  # check the top row is empty or not
        return False

    def get_next_open_row(self, column):
        for r in range(self.ROW_COUNT):
            if self.board[r][column] == 0:
                return r

    def print_board(self):
        # print(np.flip(self.board, 0), '\n')
        for row in np.flip(self.board, 0):
            print("|", end=" ")
            for col in row:
                if col == 0:
                    print(" ", end=" | ")
                elif col == 1:
                    print("1", end=" | ")
                else:
                    print("2", end=" | ")
            print("")
        print()
        time.sleep(0.1)

    def is_winner(self, piece):
        # check all the horizontal locations
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece \
                        and self.board[r][c + 2] == piece and self.board[r][c + 3] == piece:
                    return True

        # check all the vertical locations
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece \
                        and self.board[r + 2][c] == piece and self.board[r + 3][c] == piece:
                    return True

        # check all the positively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece \
                        and self.board[r + 2][c + 2] == piece and self.board[r + 3][c + 3] == piece:
                    return True

        # check all the negatively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece \
                        and self.board[r - 2][c + 2] == piece and self.board[r - 3][c + 3] == piece:
                    return True

    def is_gonna_win(self, column, piece):
        row = self.get_next_open_row(column)
        board = deepcopy(self.board)
        board[row][column] = piece

        # check all the horizontal locations
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece \
                        and board[r][c + 2] == piece and board[r][c + 3] == piece:
                    return True

        # check all the vertical locations
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c] == piece \
                        and board[r + 2][c] == piece and board[r + 3][c] == piece:
                    return True

        # check all the positively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c + 1] == piece \
                        and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                    return True

        # check all the negatively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece \
                        and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                    return True

    def num_available_moves(self):
        available_moves = []
        for c in range(self.COLUMN_COUNT):
            if self.board[self.ROW_COUNT - 1][c] == 0:
                available_moves.append(c)
        return len(available_moves)

    def get_available_moves(self):
        available_moves = []
        for c in range(self.COLUMN_COUNT):
            if self.board[self.ROW_COUNT - 1][c] == 0:
                available_moves.append(c)
        return available_moves

    def is_available_move(self):
        for c in range(self.COLUMN_COUNT):
            if self.board[self.ROW_COUNT - 1][c] == 0:
                return True
        return False

    def is_first_drop(self):
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] != 0:
                    return False
        return True

    def is_terminal_node(self):
        return self.is_winner(1) or self.is_winner(2) or self.num_available_moves() == 0

    def score_position(self, piece):
        WINDOW_LENGTH = 4
        score = 0

        # Score center column
        center_array = [int(i) for i in list(self.board[:, self.COLUMN_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(self.ROW_COUNT):
            row_array = [int(i) for i in list(self.board[r, :])]
            for c in range(self.COLUMN_COUNT - 3):
                window = row_array[c:c + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(self.COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:, c])]
            for r in range(self.ROW_COUNT - 3):
                window = col_array[r:r + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score positive sloped diagonal
        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [self.board[r + i][c + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [self.board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    @staticmethod
    def evaluate_window(window, piece):
        score = 0
        opp_piece = 2
        if piece == 2:
            opp_piece = 1

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4

        return score


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
        column = None
        while not valid_position:
            try:
                column = int(input(f'Player {self.player}, draw piece (0-6): '))
                if not game.is_valid_location(column):
                    raise ValueError
                valid_position = True
            except ValueError:
                print('Invalid column. Try again.')
        return column


class DefaultComputerPlayer(Player):
    def __init__(self, player):
        super().__init__(player)

    def get_move(self, game):
        if game.is_first_drop():
            column = random.choice(game.get_available_moves())
        else:
            column = self.default(game)['column']
        return column

    def default(self, game):
        self_player = self.player
        opponent_player = 1 if self.player == 2 else 2

        # first check if the previous move is a winner
        if game.winner == opponent_player:
            return {'column': None}
        # check if there is available move
        elif not game.is_available_move():
            return {'column': None}

        for c in game.get_available_moves():
            # Check if any move can immediately win the game
            if game.is_gonna_win(c, self_player):
                return {'column': c}

            # Check if any move can immediately block the other player from winning
            if game.is_gonna_win(c, opponent_player):
                return {'column': c}

        return {'column': random.choice(game.get_available_moves())}


class MinimaxComputerPlayer(Player):
    def __init__(self, player):
        super().__init__(player)

    def get_move(self, game):
        if game.is_first_drop():
            column = random.choice(game.get_available_moves())
        else:
            column = self.minimax(game, 5, -math.inf, math.inf, True)['column']
        return column

    def minimax(self, state, depth, alpha, beta, maximizingPlayer):
        valid_locations = state.get_available_moves()
        is_terminal = state.is_terminal_node()
        if depth == 0 or is_terminal:
            if is_terminal:
                if state.is_winner(1):
                    return {'column': None, 'score': 100000000000000}
                elif state.is_winner(2):
                    return {'column': None, 'score': -100000000000000}
                else:  # Game is over, no more valid moves
                    return {'column': None, 'score': 0}
            else:  # Depth is zero
                return {'column': None, 'score': state.score_position(1)}

        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = state.get_next_open_row(col)
                state.board[row][col] = 1
                new_score = self.minimax(state, depth - 1, alpha, beta, False)['score']
                state.board[row][col] = 0
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return {'column': column, 'score': value}

        else:  # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = state.get_next_open_row(col)
                state.board[row][col] = 2
                new_score = self.minimax(state, depth - 1, alpha, beta, True)['score']
                state.board[row][col] = 0
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return {'column': column, 'score': value}


class QlearningComputerPlayer(Player):
    def __init__(self, player, q_values):
        super().__init__(player)
        self.q_values = q_values

    def get_move(self, game):
        if game.is_first_drop():
            column = random.choice(game.get_available_moves())
        else:
            column = self.qlearning(game)['column']
        return column

    def qlearning(self, state):
        state_key = tuple(map(tuple, state.board))
        if state_key not in self.q_values:
            return {'column': random.choice(state.get_available_moves())}
        else:
            q_values = self.q_values[state_key]
            q_values = {k: v for k, v in q_values.items() if k in state.get_available_moves()}
            if not q_values:
                action = random.choice(state.get_available_moves())
            else:
                action = max(q_values.items(), key=lambda x: x[1])[0]
            return {'column': action}


class QLearning(Player):
    def __init__(self, player, epsilon=0.5, alpha=0.1, gamma=0.9):
        super().__init__(player)
        self.q_values = {}  # 存储Q值的字典
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子

    def get_move(self, game):
        if game.is_first_drop():
            column = random.choice(game.get_available_moves())
        else:
            column = self.qlearning(game)['column']
        return column

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
        return {'column': action}

    def get_q_value(self, state, action):
        if action not in self.q_values[state].keys():
            self.q_values[state][action] = 0.0
        return self.q_values[state][action]

    def update_q_table(self, game):
        winner = game.winner
        if winner == self.player:
            reward = 10000
        elif winner is None:
            reward = -100
        else:
            reward = -10000

        for state_key, action, new_state_key in game.move_history[::-1]:
            if state_key not in self.q_values:
                self.q_values[state_key] = {}
            if new_state_key not in self.q_values:
                self.q_values[new_state_key] = {}

            max_q_value = max(self.q_values[new_state_key].values()) if self.q_values[new_state_key] else 0.0
            old_q_value = self.get_q_value(state_key, action)
            new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_q_value)
            self.q_values[state_key][action] = new_q_value

            # reward = new_q_value if reward is None else self.get_q_value(state_key, action)
            reward = new_q_value


def play(game, player1, player2, print_game=True):
    if print_game:
        game.print_board()

    current_player = 1

    while game.is_available_move():
        if current_player == 1:
            column = player1.get_move(game)
        else:
            column = player2.get_move(game)

        if game.drop_piece(column, current_player):
            if print_game:
                print(f'Player {current_player} drops a piece at column {column}.')
                game.print_board()

            if game.winner:
                if print_game:
                    print(f'Player {current_player} wins!')
                return current_player  # ends the loop and exits the game
            current_player = 2 if current_player == 1 else 1  # switches player

        # time.sleep(0.1)
    if print_game:
        print('Game draw!')
    return None


def train_qlearning_connect4(epsilon, alpha, gamma, num_epochs=100000):
    player1 = QLearning(1, epsilon, alpha, gamma)
    # player2 = QLearning(2, epsilon, alpha, gamma)
    # player2 = MinimaxComputerPlayer(2)
    player2 = DefaultComputerPlayer(2)
    for i in range(num_epochs):
        game = Connect4()
        current_player = 1
        while game.is_available_move():
            # get the next move for current player
            if current_player == 2:
                column = player2.get_move(game)
            else:
                column = player1.get_move(game)
            if game.drop_piece(column, current_player):  # make the move
                if game.winner:
                    break
                current_player = 2 if current_player == 1 else 1  # switches player
        player1.update_q_table(game)
        # player2.update_q_table(game)
        if (i + 1) % (num_epochs // 100) == 0:
            print(f"Training progress: {i + 1}/{num_epochs}")

    print('num_epochs', num_epochs)
    return player1


def evaluate(rounds):
    num_rounds = rounds

    # Minimax vs Default Opponent
    minimax_vs_default_results = []
    minimax_vs_default_win = []
    minimax_vs_default_lose = []
    start_time = time.time()
    for n in range(num_rounds):
        player1 = MinimaxComputerPlayer(1)
        player2 = DefaultComputerPlayer(2)
        minimax_vs_default_result = play(Connect4(), player1, player2, False)
        minimax_vs_default_results.append(minimax_vs_default_result)
        minimax_vs_default_win.append(minimax_vs_default_results.count(player1.player) / (n+1))
        minimax_vs_default_lose.append(minimax_vs_default_results.count(player2.player) / (n+1))
    end_time = time.time()
    total_time = end_time - start_time
    plt.plot(minimax_vs_default_win, label='Rate (Minimax Win)')
    plt.plot(minimax_vs_default_lose, label='Rate (Default Opponent Win)')
    plt.title(f'Minimax vs Default Opponent, sec / round: {total_time / num_rounds:.5f}')
    plt.legend()
    plt.show()

    # Q-Learning vs Default Opponent
    with open('q_table_connect4.pkl', 'rb') as f:
        q_values_table = pickle.load(f)
    qlearning_vs_default_results = []
    qlearning_vs_default_win = []
    qlearning_vs_default_lose = []
    start_time = time.time()
    for n in range(num_rounds):
        player1 = QlearningComputerPlayer(1, q_values_table)
        player2 = DefaultComputerPlayer(2)
        qlearning_vs_default_result = play(Connect4(), player1, player2, False)
        qlearning_vs_default_results.append(qlearning_vs_default_result)
        qlearning_vs_default_win.append(qlearning_vs_default_results.count(player1.player) / (n + 1))
        qlearning_vs_default_lose.append(qlearning_vs_default_results.count(player2.player) / (n + 1))
    end_time = time.time()
    total_time = end_time - start_time
    plt.plot(qlearning_vs_default_win, label='Rate (Q-Learning Win)')
    plt.plot(qlearning_vs_default_lose, label='Rate (Default Opponent Win)')
    plt.title(f'Q-Learning vs Default Opponent, sec / round: {total_time / num_rounds:.5f}')
    plt.legend()
    plt.show()

    # Minimax vs QLearning
    minimax_vs_qlearning_results = []
    minimax_vs_qlearning_win = []
    minimax_vs_qlearning_lose = []
    start_time = time.time()
    for n in range(num_rounds):
        player1 = MinimaxComputerPlayer(1)
        player2 = QlearningComputerPlayer(2, q_values_table)
        minimax_vs_qlearning_result = play(Connect4(), player1, player2, False)
        minimax_vs_qlearning_results.append(minimax_vs_qlearning_result)
        minimax_vs_qlearning_win.append(minimax_vs_qlearning_results.count(player1.player)/(n+1))
        minimax_vs_qlearning_lose.append(minimax_vs_qlearning_results.count(player2.player)/(n+1))
    end_time = time.time()
    total_time = end_time - start_time
    plt.plot(minimax_vs_qlearning_win, label='Rate (Minimax Win)')
    plt.plot(minimax_vs_qlearning_lose, label='Rate (Q-Learning Win)')
    plt.title(f'Minimax vs Q-Learning, sec / round: {total_time / num_rounds:.5f}')
    plt.legend()
    plt.show()


def execute():
    with open('q_table_connect4.pkl', 'rb') as f:
        q_values_table = pickle.load(f)

    player1 = None
    player2 = None

    while player1 is None:
        try:
            input_player1 = int(input('Choose role for player 1 (1-Human, 2-DefaultOpponent, 3-Minimax, 4-QLearning):'))
            if input_player1 == 1:
                player1 = HumanPlayer(1)
            elif input_player1 == 2:
                player1 = DefaultComputerPlayer(1)
            elif input_player1 == 3:
                player1 = MinimaxComputerPlayer(1)
            elif input_player1 == 4:
                player1 = QlearningComputerPlayer(1, q_values_table)
            else:
                raise ValueError
        except ValueError:
            print('Please enter number from 1-4.')

    while player2 is None:
        try:
            input_player2 = int(input('Choose role for player 2 (1-Human, 2-DefaultOpponent, 3-Minimax, 4-QLearning):'))
            if input_player2 == 1:
                player2 = HumanPlayer(2)
            elif input_player2 == 2:
                player2 = DefaultComputerPlayer(2)
            elif input_player2 == 3:
                player2 = MinimaxComputerPlayer(2)
            elif input_player2 == 4:
                player2 = QlearningComputerPlayer(2, q_values_table)
            else:
                raise ValueError
        except ValueError:
            print('Please enter number from 1-4.')

    play(Connect4(), player1, player2, True)


if __name__ == '__main__':
    execute()
    evaluate(int(input('Enter the number of rounds to evaluate algorithms:')))
