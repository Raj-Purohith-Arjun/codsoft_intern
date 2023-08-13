
HUMAN = 'X'
AI = 'O'
EMPTY = ' '

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def check_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i][j] == player for j in range(3)):
            return True
        if all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]

def minimax(board, depth, maximizing_player):
    if check_winner(board, AI):
        return 1
    if check_winner(board, HUMAN):
        return -1
    if not get_empty_cells(board):
        return 0
    
    if maximizing_player:
        max_eval = float('-inf')
        for i, j in get_empty_cells(board):
            board[i][j] = AI
            eval = minimax(board, depth + 1, False)
            board[i][j] = EMPTY
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = HUMAN
            eval = minimax(board, depth + 1, True)
            board[i][j] = EMPTY
            min_eval = min(min_eval, eval)
        return min_eval

def get_best_move(board):
    best_move = None
    best_eval = float('-inf')
    for i, j in get_empty_cells(board):
        board[i][j] = AI
        eval = minimax(board, 0, False)
        board[i][j] = EMPTY
        if eval > best_eval:
            best_eval = eval
            best_move = (i, j)
    return best_move

def main():
    board = [[EMPTY, EMPTY, EMPTY] for _ in range(3)]
    print_board(board)
    
    while True:
        human_move = None
        while human_move not in get_empty_cells(board):
            row = int(input("Enter row (0, 1, or 2): "))
            col = int(input("Enter column (0, 1, or 2): "))
            human_move = (row, col)
        
        board[human_move[0]][human_move[1]] = HUMAN
        print_board(board)
        
        if check_winner(board, HUMAN):
            print("Human wins!")
            break
        elif not get_empty_cells(board):
            print("It's a draw!")
            break
        
        ai_move = get_best_move(board)
        board[ai_move[0]][ai_move[1]] = AI
        print("AI's move:")
        print_board(board)
        
        if check_winner(board, AI):
            print("AI wins!")
            break
        elif not get_empty_cells(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
