from Chessnut import Game
import random

def prioritize_checkmate(game, moves):
    """
    Prioritize moves that result in checkmate.

    Args:
        game: The current Game object.
        moves: List of legal moves.

    Returns:
        A move that results in checkmate, or None if none found.
    """
    for move in moves[:10]:  # Check only a subset for efficiency
        temp_game = Game(game.get_fen())
        temp_game.apply_move(move)
        if temp_game.status == Game.CHECKMATE:
            return move
    return None

def prioritize_captures(game, moves):
    """
    Prioritize moves that capture opponent pieces.

    Args:
        game: The current Game object.
        moves: List of legal moves.

    Returns:
        A move that captures an opponent piece, or None if none found.
    """
    for move in moves:
        target_square = move[2:4]
        if game.board.get_piece(Game.xy2i(target_square)) != ' ':
            return move
    return None

def prioritize_queen_promotion(moves):
    """
    Prioritize moves that result in a queen promotion.

    Args:
        moves: List of legal moves.

    Returns:
        A move that results in a queen promotion, or None if none found.
    """
    for move in moves:
        if "q" in move.lower():
            return move
    return None

def bot(obs):
    """
    Modular chess bot that prioritizes checkmates, captures, queen promotions, then randomly moves.

    Args:
        obs: An object with a 'board' attribute representing the current board state as a FEN string.

    Returns:
        A string representing the chosen move in UCI notation (e.g., "e2e4")
    """
    game = Game(obs.board)
    moves = list(game.get_moves())

    # 1. Check for checkmates
    move = prioritize_checkmate(game, moves)
    if move:
        return move

    # 2. Check for captures
    move = prioritize_captures(game, moves)
    if move:
        return move

    # 3. Check for queen promotions
    move = prioritize_queen_promotion(moves)
    if move:
        return move

    # 4. Random move as fallback
    return random.choice(moves)