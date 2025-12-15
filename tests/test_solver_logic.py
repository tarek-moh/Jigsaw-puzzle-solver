import sys
import os
import numpy as np

# Add parent directory to path to import solver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import solver

def test_jigsaw_logic():
    print("Testing Jigsaw Logic...")
    
    # -------------------------------------------------------------
    # INDEXING SPECIFICATION
    # Score matrix stores 4*N entries.
    # Order: Piece 0, Piece 1, ...
    # Each piece has 4 indices: 
    #   idx + 0: Top
    #   idx + 1: Right
    #   idx + 2: Bottom
    #   idx + 3: Left
    # -------------------------------------------------------------
    
    TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
    SIDES_PER_PIECE = 4

    # === Test 1: 2x1 Puzzle (Vertical) ===
    # Piece 0 on Top, Piece 1 on Bottom
    # Match: Piece 0 Bottom <-> Piece 1 Top
    
    num_pieces = 2
    total_strips = num_pieces * SIDES_PER_PIECE
    score_matrix = np.zeros((total_strips, total_strips))
    
    # Indices
    p0_bottom = 0 * SIDES_PER_PIECE + BOTTOM
    p1_top    = 1 * SIDES_PER_PIECE + TOP
    
    # Set strong match
    score_matrix[p0_bottom, p1_top] = 0.95
    score_matrix[p1_top, p0_bottom] = 0.95
    
    print("\nRunning solve_jigsaw_greedy on 2x1 puzzle...")
    result_pieces = solver.solve_jigsaw_greedy(score_matrix, num_pieces)
    print("Result Pieces:", result_pieces)
    
    # Invert to get piece -> pos
    pos_map = {pid: pos for pos, pid in result_pieces.items()}
    
    p0_pos = pos_map[0]
    p1_pos = pos_map[1]
    
    diff_r = p1_pos[0] - p0_pos[0]
    diff_c = p1_pos[1] - p0_pos[1]
    
    if diff_r == 1 and diff_c == 0:
        print("PASS: Piece 1 is correctly placed below Piece 0.")
    else:
        print(f"FAIL: Expected delta (1, 0), got ({diff_r}, {diff_c})")

    # === Test 2: 2x2 Puzzle ===
    # 0 1
    # 2 3
    
    print("\nTesting 2x2 Puzzle matching...")
    num_pieces = 4
    total_strips = num_pieces * SIDES_PER_PIECE
    score_matrix = np.zeros((total_strips, total_strips))
    
    # Define matches explicitly using constants
    matches = [
        # Piece 0 Right <-> Piece 1 Left
        (0*SIDES_PER_PIECE + RIGHT, 1*SIDES_PER_PIECE + LEFT),
        
        # Piece 0 Bottom <-> Piece 2 Top
        (0*SIDES_PER_PIECE + BOTTOM, 2*SIDES_PER_PIECE + TOP),
        
        # Piece 1 Bottom <-> Piece 3 Top
        (1*SIDES_PER_PIECE + BOTTOM, 3*SIDES_PER_PIECE + TOP),
        
        # Piece 2 Right <-> Piece 3 Left
        (2*SIDES_PER_PIECE + RIGHT,  3*SIDES_PER_PIECE + LEFT)
    ]
    
    for s1, s2 in matches:
        score_matrix[s1, s2] = 0.9
        score_matrix[s2, s1] = 0.9
        
    result_pieces = solver.solve_jigsaw_greedy(score_matrix, num_pieces)
    print("Result Pieces 2x2:", result_pieces)
    
    pos_map_2x2 = {pid: pos for pos, pid in result_pieces.items()}
    
    r0, c0 = pos_map_2x2[0]
    r1, c1 = pos_map_2x2[1]
    r2, c2 = pos_map_2x2[2]
    r3, c3 = pos_map_2x2[3]
    
    if (r1 == r0 and c1 == c0 + 1) and \
       (r2 == r0 + 1 and c2 == c0) and \
       (r3 == r0 + 1 and c3 == c0 + 1):
        print("PASS: 2x2 Arrangement correct.")
    else:
        print("FAIL: 2x2 Arrangement incorrect.")

if __name__ == "__main__":
    test_jigsaw_logic()
