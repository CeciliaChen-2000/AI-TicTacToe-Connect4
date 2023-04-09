import pickle
from TicTacToe import train_qlearning_tictactoe, QlearningComputerPlayer, DefaultComputerPlayer, TicTacToe, play


def choose_parameters():
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    biggest_winning_rate = 0
    smallest_losing_rate = 1
    best_e = 0.0
    best_a = 0.0

    for e in epsilons:
        for a in alphas:
            train_player = train_qlearning_tictactoe(epsilon=e, alpha=a, gamma=0.9, num_epochs=10000)
            player1 = QlearningComputerPlayer('X', train_player.q_values)
            player2 = DefaultComputerPlayer('O')

            results = []
            for _ in range(1000):
                results.append(play(TicTacToe(), player1, player2, print_game=False))
            current_winning_rate = results.count(player1.player) / len(results)
            current_losing_rate = results.count(player2.player) / len(results)

            if current_winning_rate > biggest_winning_rate and current_losing_rate < smallest_losing_rate:
                biggest_winning_rate = current_winning_rate
                smallest_losing_rate = current_losing_rate
                best_e = e
                best_a = a
    return best_e, best_a


def save_best_q_table(e, a):
    biggest_winning_rate = 0
    smallest_losing_rate = 1
    best_q_values = {}
    for i in range(1000):
        train_player = train_qlearning_tictactoe(epsilon=e, alpha=a, gamma=0.9, num_epochs=10000)

        player1 = QlearningComputerPlayer('X', train_player.q_values)
        player2 = DefaultComputerPlayer('O')

        results = []
        for _ in range(1000):
            results.append(play(TicTacToe(), player1, player2, print_game=False))
        current_winning_rate = results.count(player1.player) / len(results)
        current_losing_rate = results.count(player2.player) / len(results)

        if current_winning_rate > biggest_winning_rate and current_losing_rate < smallest_losing_rate:
            biggest_winning_rate = current_winning_rate
            smallest_losing_rate = current_losing_rate
            best_q_values = player1.q_values

    with open('q_table_tictactoe.pkl', 'wb') as f:
        pickle.dump(best_q_values, f)
    return biggest_winning_rate, smallest_losing_rate


epsilon, alpha = choose_parameters()
print(save_best_q_table(epsilon, alpha))
