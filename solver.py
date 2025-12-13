import numpy as np
import heapq

def normalize_ssd(ssd_matrix):
    """
    Normalizes SSD errors to a similarity score [0, 1].
    SSD is a dissimilarity metric (0 is best).
    We invert it so 1 is best.
    Formula: 1 - (x / max(x))
    """
    ssd_matrix = np.array(ssd_matrix)
    if ssd_matrix.size == 0:
        return ssd_matrix
    
    max_val = np.max(ssd_matrix)
    if max_val == 0:
        # If all errors are 0, then all are perfect matches (score 1)
        return np.ones_like(ssd_matrix, dtype=float)
        
    return 1.0 - (ssd_matrix / max_val)

def normalize_correlation(corr_matrix):
    """
    Normalizes correlation coefficients [-1, 1] to a similarity score [0, 1].
    Correlation is a similarity metric (1 is best).
    Formula: (x + 1) / 2
    """
    corr_matrix = np.array(corr_matrix)
    # Ensure it's within expected bounds just in case
    return (np.clip(corr_matrix, -1.0, 1.0) + 1.0) / 2.0

def normalize_gradient(grad_matrix):
    """
    Normalizes gradient errors to a similarity score [0, 1].
    Gradient error is a dissimilarity metric (0 is best).
    Formula: 1 - (x / max(x))
    """
    grad_matrix = np.array(grad_matrix)
    if grad_matrix.size == 0:
        return grad_matrix
        
    max_val = np.max(grad_matrix)
    if max_val == 0:
        return np.ones_like(grad_matrix, dtype=float)
        
    return 1.0 - (grad_matrix / max_val)

def calculate_composite_score(metrics, weights=None):
    """
    Calculates the weighted sum of normalized metrics.
    
    Args:
        metrics (dict): Dictionary where keys are metric names and values are 
                        normalized score matrices (numpy arrays).
                        e.g., {'ssd': ssd_scores, 'grad': grad_scores}
        weights (dict): Dictionary of weights for each metric. 
                        e.g., {'ssd': 1.0, 'grad': 2.0}
                        If None, equal weights are used.
                        
    Returns:
        numpy.ndarray: The combined score matrix.
    """
    if not metrics:
        return None
        
    # Get shape from the first metric
    first_key = list(metrics.keys())[0]
    shape = metrics[first_key].shape
    
    total_score = np.zeros(shape)
    total_weight = 0.0
    
    for name, matrix in metrics.items():
        weight = 1.0
        if weights and name in weights:
            weight = weights[name]
            
        total_score += matrix * weight
        total_weight += weight
        
    if total_weight == 0:
        return np.zeros(shape)
        
    return total_score / total_weight

# ==========================================
# Ordering Logic Implementation
# ==========================================

def find_best_buddies(score_matrix):
    """
    Identifies mutual best matches ("Best Buddies") from the score matrix.
    
    Args:
        score_matrix (numpy.ndarray): (4*N) x (4*N) matrix of similarity scores.
                                      Index i represents strip i.
                                      score_matrix[i, j] is match score between strip i (of A) and strip j (of B).
                                      
    Returns:
        list: List of tuples (strip_idx_a, strip_idx_b, score).
    """
    num_strips = score_matrix.shape[0]
    
    # 1. Find best match for each strip
    # best_matches[i] = (best_j, score)
    best_matches = {}
    
    for i in range(num_strips):
        piece_idx = i // 4
        side_idx = i % 4
        target_side = (side_idx + 2) % 4 # Opposite side for standard orientation matches
        
        best_j = -1
        best_score = -1.0
        
        # Iterate to find best match for strip i on the target side
        for j in range(num_strips):
            j_piece = j // 4
            j_side = j % 4
            
            if piece_idx == j_piece:
                continue # Skip same piece
                
            if j_side == target_side:
                # This is a valid geometric candidate
                score = score_matrix[i, j]
                if score > best_score:
                    best_score = score
                    best_j = j
                    
        if best_j != -1:
            best_matches[i] = (best_j, best_score)
            
    # 2. Check for mutuality
    best_buddies = []
    seen_pairs = set()
    
    for i, (best_j, score_i) in best_matches.items():
        if best_j in best_matches:
            # Check if j's best match is i
            j_best_match_idx, score_j = best_matches[best_j]
            
            if j_best_match_idx == i:
                # Mutual Match!
                # Store as sorted tuple to avoid duplication
                pair = tuple(sorted((i, best_j)))
                if pair not in seen_pairs:
                    # Average the scores
                    avg_score = (score_i + score_j) / 2
                    best_buddies.append((pair[0], pair[1], avg_score))
                    seen_pairs.add(pair)
                    
    return best_buddies

class JigsawCluster:
    def __init__(self, piece_id):
        self.pieces = {} # (row, col) -> piece_id
        self.pieces[(0, 0)] = piece_id
        self.id_to_pos = {piece_id: (0, 0)} # piece_id -> (row, col)
        
    def get_relative_position(self, piece_id):
        return self.id_to_pos.get(piece_id)
        
    def is_compatible(self, other_cluster, relative_r, relative_c):
        """
        Check if merging other_cluster at (relative_r, relative_c) would cause overlaps.
        relative_r, relative_c is where (0,0) of other_cluster would land in this cluster's coords.
        """
        for (r, c), pid in other_cluster.pieces.items():
            new_r = r + relative_r
            new_c = c + relative_c
            if (new_r, new_c) in self.pieces:
                return False # Overlap!
        return True
        
    def merge(self, other_cluster, relative_r, relative_c):
        """
        Merges other_cluster into this one.
        """
        for (r, c), pid in other_cluster.pieces.items():
            new_r = r + relative_r
            new_c = c + relative_c
            self.pieces[(new_r, new_c)] = pid
            self.id_to_pos[pid] = (new_r, new_c)

def solve_jigsaw_greedy(score_matrix, num_pieces):
    """
    Solves the jigsaw puzzle using a greedy best-buddy approach.
    
    Args:
        score_matrix (numpy.ndarray): Combined score matrix.
        num_pieces (int): Total number of pieces.
        
    Returns:
        dict: Final cluster configuration { (row, col) : piece_id }
              Ideally should return one large cluster.
    """
    
    # 1. Initialize clusters
    clusters = {pid: JigsawCluster(pid) for pid in range(num_pieces)}
    piece_to_cluster = {pid: pid for pid in range(num_pieces)} # Map piece_id to cluster_id (initially same)
    
    # 2. Get Best Buddies
    buddies = find_best_buddies(score_matrix)
    
    # 3. Create Priority Queue
    # We want max score, so perform neg score for heapq (min heap)
    pq = []
    for s1, s2, score in buddies:
        # Pushing negative score for max-heap behavior
        heapq.heappush(pq, (-score, s1, s2))
        
    # 4. Greedy Assembly
    while pq:
        neg_score, s1_idx, s2_idx = heapq.heappop(pq)
        score = -neg_score
        
        # Identify Pieces and Clusters
        p1 = s1_idx // 4
        p2 = s2_idx // 4
        
        c1_id = piece_to_cluster[p1]
        c2_id = piece_to_cluster[p2]
        
        if c1_id == c2_id:
            # Already in same cluster. 
            continue
            
        cluster1 = clusters[c1_id]
        cluster2 = clusters[c2_id]
        
        # Determine geometric relation
        # We know s1 and s2 match.
        s1_side = s1_idx % 4
        s2_side = s2_idx % 4
        
        # Piece 1 pos in Cluster 1
        r1, c1 = cluster1.get_relative_position(p1)
        
        # Piece 2 pos in Cluster 2 (currently)
        r2_local, c2_local = cluster2.get_relative_position(p2)
        
        # Target position of Piece 2 relative to Piece 1
        # P1 Top(0) matches P2 Bottom(2) -> P2 is Above P1 (r1-1, c1)
        # P1 Right(1) matches P2 Left(3) -> P2 is Right of P1 (r1, c1+1)
        # P1 Bottom(2) matches P2 Top(0) -> P2 is Below P1 (r1+1, c1)
        # P1 Left(3) matches P2 Right(1) -> P2 is Left of P1 (r1, c1-1)
        
        dr, dc = 0, 0
        if s1_side == 0: # Top
            dr, dc = -1, 0
        elif s1_side == 1: # Right
            dr, dc = 0, 1
        elif s1_side == 2: # Bottom
            dr, dc = 1, 0
        elif s1_side == 3: # Left
            dr, dc = 0, -1
            
        target_r2_in_c1 = r1 + dr
        target_c2_in_c1 = c1 + dc
        
        # Shift required for Cluster 2:
        # new_pos = old_pos + shift
        # target_pos = r2_local + shift
        # shift = target_pos - r2_local
        
        shift_r = target_r2_in_c1 - r2_local
        shift_c = target_c2_in_c1 - c2_local
        
        # Check Compatibility
        if cluster1.is_compatible(cluster2, shift_r, shift_c):
            # MERGE!
            cluster1.merge(cluster2, shift_r, shift_c)
            
            # Update pointers
            # All pieces in c2 need to point to c1_id
            for pid in clusters[c2_id].id_to_pos.keys():
                piece_to_cluster[pid] = c1_id
                
            # Remove c2 from valid clusters
            del clusters[c2_id]
            
    # Return the largest cluster (hopefully the only one)
    if not clusters:
        return {}
        
    largest_cluster = max(clusters.values(), key=lambda c: len(c.pieces))
    return largest_cluster.pieces
