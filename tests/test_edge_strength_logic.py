import sys
import os
import numpy as np

# Add parent directory to path to import solver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Starting import of solver...")
import solver
print("Import of solver complete.")

def test_low_edge_strength_suppression():
    """
    Simulates a scenario where:
    - Match (A, B) is correct (High Similarity Score) but has Low Edge Strength.
    - Match (A, C) is incorrect (Low Similarity Score) but has High Edge Strength.
    
    Current Logic Prediction:
    - Score(A, B) = 0.9 * 0.1 = 0.09
    - Score(A, C) = 0.2 * 1.0 = 0.20
    -> Solver incorrectly chooses C over B.
    
    Desired Logic:
    - Solver should prioritize the strong similarity of (A, B) despite the low edge strength.
    """
    print("\n=== Testing Low Edge Strength Suppression ===")
    
    # Setup: 3 pieces, we focus on Piece 0 Side Right (1) matching Piece 1 Side Left (3)
    num_pieces = 3
    total_strips = num_pieces * 4
    score_matrix = np.zeros((total_strips, total_strips))
    edge_strengths = np.zeros(total_strips)
    
    # Indices
    # P0 Right
    p0_right = 0 * 4 + 1
    # P1 Left (Correct Match)
    p1_left = 1 * 4 + 3
    # P2 Left (Incorrect Match)
    p2_left = 2 * 4 + 3
    
    # 1. Define Raw Scores (SSD/Corr combined)
    # Strong match between P0 and P1
    score_matrix[p0_right, p1_left] = 0.95
    score_matrix[p1_left, p0_right] = 0.95
    
    # Weak match between P0 and P2
    score_matrix[p0_right, p2_left] = 0.3
    score_matrix[p2_left, p0_right] = 0.3
    
    # 2. Define Edge Strengths
    # P0 and P1 are completely flat (sky) -> 0.0 strength
    edge_strengths[p0_right] = 0.0
    edge_strengths[p1_left] = 0.0
    # P2 is also flat or whatever
    edge_strengths[p2_left] = 0.0
    
    # Run find_best_buddies ONLY (to isolate scoring logic)
    print(f"Edge Strengths: P0_R={edge_strengths[p0_right]}, P1_L={edge_strengths[p1_left]}, P2_L={edge_strengths[p2_left]}")
    print(f"Raw Scores: P0-P1={score_matrix[p0_right, p1_left]}, P0-P2={score_matrix[p0_right, p2_left]}")
    
    best_buddies = solver.find_best_buddies(score_matrix, edge_strengths)
    
    # Analyze results
    print("Best Buddies Found:", best_buddies)
    
    # We expect a pair (start_idx, end_idx, score)
    # Check if P0 Right is paired with P1 Left
    # best_buddies stores (idx1, idx2, score)
    
    found_match = False
    for s1, s2, score in best_buddies:
        # Check for (p0_right, p1_left) or (p1_left, p0_right)
        if (s1 == p0_right and s2 == p1_left) or (s1 == p1_left and s2 == p0_right):
            found_match = True
            print(f"PASS: Correctly matched P0 and P1 despite low edge strength. Score: {score}")
            break
            
    if not found_match:
        print("FAIL: Failed to match P0 and P1. Likely suppressed by edge strength.")
        # Check if it matched P2 instead?
        # Manually verify calculation
        score_p0_p1 = 0.95 * 0.1
        score_p0_p2 = 0.3 * 0.1 # wait, formula is min(edge[i], edge[j])
        # P0-P1: 0.95 * min(0.1, 0.1) = 0.095
        # P0-P2: 0.3 * min(0.1, 0.9) = 0.3 * 0.1 = 0.03
        
        # Consider a case where P2 is the spoiler
        # Assume there's a P3 that matches P2 strongly but P2 prefers P0?
        # This is hard to construct purely with one pair.
        
        # Let's adjust the test case to be more provocative if this passes.
        # But honestly, if P0-P1 is a mutual best match in raw score,
        # and edge strength just scales it, it should remain mutual best match unless
        # something else overtakes it.
        
        # The real issue might be that the score drops so low that it's considered "garbage" or
        # maybe it falls below the score of a non-match that happens to have high edge strength?
        # Wait, if s(i,j) = raw(i,j) * min(e_i, e_j)
        # s(0,1) = 0.95 * 0.1 = 0.095
        # s(0,2) = 0.3 * 0.1 = 0.03
        # s(2,0) = 0.3 * 0.1 = 0.03
        
        # What if there is another piece P3?
        # s(2,3) = 0.2 * 0.9 = 0.18
        # P2 prefers P3 (0.18) over P0 (0.03).
        # P0 prefers P1 (0.095).
        # P1 prefers P0 (0.095).
        
        # Ideally P0-P1 should be a buddy.
        # So why are clusters small?
        # Maybe the scores are just too small generally?
        
        pass

if __name__ == "__main__":
    test_low_edge_strength_suppression()
