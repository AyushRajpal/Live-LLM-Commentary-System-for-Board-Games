
def coord_to_notation(row, col):
    """Convert (row, col) coordinates to chess-like notation (e.g., A1, B2)."""
    col_letter = chr(65 + col)  # A, B, C, ...
    row_number = str(row + 1)   # 1, 2, 3, ...
    return col_letter + row_number

def notation_to_coord(notation):
    """Convert chess-like notation (e.g., A1, B2) to (row, col) coordinates."""
    col = ord(notation[0]) - 65  # A -> 0, B -> 1, ...
    row = int(notation[1]) - 1   # 1 -> 0, 2 -> 1, ...
    return row, col

def format_move(move):
    """Format a move from ((row1, col1), (row2, col2)) to 'A1 to B2'."""
    if not move:
        return "no move available"
    start, end = move
    start_notation = coord_to_notation(start[0], start[1])
    end_notation = coord_to_notation(end[0], end[1])
    return f"{start_notation} to {end_notation}"