from Chessnut import Game
import random

def prioritize_checkmate(game, moves):
    """
    Prioritize moves that result in checkmate, checking up to a specified depth recursively.

    Args:
        game: The current Game object.
        moves: List of legal moves.
        depth: Maximum depth to search for a checkmate.

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

def prioritize_center_control(game, moves):
    """
    Prioritize moves that control the center squares and the squares around them.

    Args:
        game: The current Game object.
        moves: List of legal moves.

    Returns:
        A move that controls the center or surrounding squares, or None if none found.
    """
    center_squares = {"e4", "d4", "e5", "d5", "c3", "c4", "c5", "c6", "d3", "e3", "f3", "f4", "f5", "f6", "d6", "e6"}
    for move in moves:
        target_square = move[2:4]
        if target_square in center_squares:
            return move
    return None

def prioritize_development(game, moves):
    """
    Prioritize moves that develop pieces towards active positions (e.g., knights, bishops), accounting for color.

    Args:
        game: The current Game object.
        moves: List of legal moves.

    Returns:
        A move that develops a piece, or None if none found.
    """
    development_squares_white = {"c3", "f3", "c2", "f2", "b3", "g3", "b2", "g2"}
    development_squares_black = {"c6", "f6", "c7", "f7", "b6", "g6", "b7", "g7"}
    is_white = game.board.turn == "w"
    development_squares = development_squares_white if is_white else development_squares_black

    for move in moves:
        target_square = move[2:4]
        if target_square in development_squares:
            return move
    return None

def chess_bot(obs):
    """
    Modular chess bot that prioritizes checkmates, captures, queen promotions, center control, development, then randomly moves.

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

    # 4. Check for center control
    move = prioritize_center_control(game, moves)
    if move:
        return move

    # 5. Check for development
    move = prioritize_development(game, moves)
    if move:
        return move

    # 6. Random move as fallback
    return random.choice(moves)
