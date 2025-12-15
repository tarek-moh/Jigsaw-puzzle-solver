import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from scipy.ndimage import sobel
import os

# ==========================================
# Image Processing & Utilities
# ==========================================

def readImage(path):
    if not os.path.exists(path):
        print(f"Image not found at {path}")
        return None
    return cv2.imread(path)

def showImage(img, pltx=5, plty=5, title="Image"):
    if img is None:
        print("Failed to load image.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(pltx, plty))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

def divide_image(image, number):
    if not math.isqrt(number) == math.sqrt(number):
        raise ValueError("The 'number' of pieces must be a perfect square.")
    
    grid_size = int(math.sqrt(number))
    pieces = []
    height, width, _ = image.shape
    
    piece_height = height // grid_size
    piece_width = width // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * piece_height
            y_end = (i + 1) * piece_height
            x_start = j * piece_width
            x_end = (j + 1) * piece_width
            
            piece = image[y_start:y_end, x_start:x_end]
            pieces.append(piece)
    return pieces

def extract_strips(image, border=1):
    """
    Returns unified edge strips with shape (length, channels), all vertical.
    Order: [top, right, bottom, left]
    """
    # Extract raw strips
    top = image[:border, :, :]       # (b, W, C)
    bottom = image[-border:, :, :]   # (b, W, C)
    left = image[:, :border, :]      # (H, b, C)
    right = image[:, -border:, :]    # (H, b, C)

    # Rotate top/bottom to become vertical
    bottom = np.rot90(bottom, k=1)   # rotate 90deg counter-clockwise
    top = np.rot90(top, k=1)         # rotate 90deg counter-clockwise
    
    # Flip to ensure matching orientation
    bottom = np.flip(bottom, axis=1)   
    right = np.flip(right, axis=1)   

    return [top, right, bottom, left]

def get_all_strips(pieces, border=1):
    all_strips = []
    for piece in pieces:
        strips = extract_strips(piece, border)
        all_strips.append(strips)
    return all_strips

# ==========================================
# Metric Calculations
# ==========================================

def diff_squared(strip1, strip2):
    strip1 = np.asarray(strip1, dtype=np.float32)
    strip2 = np.asarray(strip2, dtype=np.float32)
    return np.sum((strip1 - strip2) ** 2)

def gradient_error(strip1, strip2):
    # If color image, convert to grayscale
    if strip1.ndim == 3:
        strip1_gray = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
        strip2_gray = cv2.cvtColor(strip2, cv2.COLOR_BGR2GRAY)
    else:
        strip1_gray = strip1
        strip2_gray = strip2

    grad1 = sobel(strip1_gray, axis=0)
    grad2 = sobel(strip2_gray, axis=0)
    
    return np.mean((grad1 - grad2)**2)

def z_score(strip):
    if len(strip.shape) == 3:
        strip_gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    else:
        strip_gray = strip.copy()
        
    strip_flat = strip_gray.flatten()
    mean = np.mean(strip_flat)
    std = np.std(strip_flat)
    if std == 0:
        std = 1e-6
    return (strip_flat - mean) / std

def compute_corr_error(stripA, stripB):
    zA = z_score(stripA)
    zB = z_score(stripB)
    
    min_len = min(len(zA), len(zB))
    zA = zA[:min_len]
    zB = zB[:min_len]
    
    # We only need correlation for the score matrix according to previous logic
    correlation = np.sum(zA * zB) / min_len
    return correlation

# ==========================================
# Normalization & Scoring
# ==========================================

def normalize_ssd(ssd_matrix):
    ssd_matrix = np.array(ssd_matrix)
    if ssd_matrix.size == 0: return ssd_matrix
    max_val = np.max(ssd_matrix)
    if max_val == 0: return np.ones_like(ssd_matrix, dtype=float)
    return 1.0 - (ssd_matrix / max_val)

def normalize_correlation(corr_matrix):
    corr_matrix = np.array(corr_matrix)
    return (np.clip(corr_matrix, -1.0, 1.0) + 1.0) / 2.0

def normalize_gradient(grad_matrix):
    grad_matrix = np.array(grad_matrix)
    if grad_matrix.size == 0: return grad_matrix
    max_val = np.max(grad_matrix)
    if max_val == 0: return np.ones_like(grad_matrix, dtype=float)
    return 1.0 - (grad_matrix / max_val)

def calculate_composite_score(metrics, weights=None):
    if not metrics: return None
    first_key = list(metrics.keys())[0]
    shape = metrics[first_key].shape
    total_score = np.zeros(shape)
    total_weight = 0.0
    for name, matrix in metrics.items():
        weight = weights.get(name, 1.0) if weights else 1.0
        total_score += matrix * weight
        total_weight += weight
    if total_weight == 0: return np.zeros(shape)
    return total_score / total_weight

def calculate_all_metrics(all_strips):
    """
    Computes SSD, Correlation, and Gradient score matrices for all strips.
    Returns composite score matrix.
    Index i = Piece_Idx * 4 + Side_Idx
    """
    num_pieces = len(all_strips)
    num_sides = 4
    total_strips = num_pieces * num_sides
    
    flat_strips = []
    for p_strips in all_strips:
        flat_strips.extend(p_strips)
        
    ssd_mat = np.zeros((total_strips, total_strips))
    corr_mat = np.zeros((total_strips, total_strips))
    grad_mat = np.zeros((total_strips, total_strips))
    
    for i in range(total_strips):
        for j in range(total_strips):
            if i // 4 == j // 4: # Same piece
                continue
                
            s1 = flat_strips[i]
            s2 = flat_strips[j]
            
            ssd_mat[i, j] = diff_squared(s1, s2)
            corr_mat[i, j] = compute_corr_error(s1, s2)
            grad_mat[i, j] = gradient_error(s1, s2)
            
    norm_ssd = normalize_ssd(ssd_mat)
    norm_corr = normalize_correlation(corr_mat)
    norm_grad = normalize_gradient(grad_mat)
    
    weights = {'ssd': .9, 'corr': .05, 'grad': .05}
    combined = calculate_composite_score({
        'ssd': norm_ssd,
        'corr': norm_corr,
        'grad': norm_grad 
    }, weights)
    
    return combined

# ==========================================
# Solver Logic
# ==========================================

def find_best_buddies(score_matrix):
    num_strips = score_matrix.shape[0]
    best_matches = {}
    
    for i in range(num_strips):
        piece_idx = i // 4
        side_idx = i % 4
        target_side = (side_idx + 2) % 4
        
        best_j = -1
        best_score = -1.0
        
        for j in range(num_strips):
            j_piece = j // 4
            j_side = j % 4
            if piece_idx == j_piece: continue
            
            if j_side == target_side:
                score = score_matrix[i, j]
                if score > best_score:
                    best_score = score
                    best_j = j
                    
        if best_j != -1:
            best_matches[i] = (best_j, best_score)
            
    best_buddies = []
    seen = set()
    for i, (best_j, score_i) in best_matches.items():
        if best_j in best_matches:
            j_match, score_j = best_matches[best_j]
            if j_match == i:
                pair = tuple(sorted((i, best_j)))
                if pair not in seen:
                    avg_score = (score_i + score_j) / 2
                    best_buddies.append((pair[0], pair[1], avg_score))
                    seen.add(pair)
    return best_buddies

class JigsawCluster:
    def __init__(self, piece_id):
        self.pieces = { (0,0): piece_id }
        self.id_to_pos = { piece_id: (0,0) }
        
    def get_relative_position(self, piece_id):
        return self.id_to_pos.get(piece_id)
        
    def is_compatible(self, other, shift_r, shift_c):
        for (r,c), pid in other.pieces.items():
            if (r + shift_r, c + shift_c) in self.pieces:
                return False
        return True
        
    def merge(self, other, shift_r, shift_c):
        for (r,c), pid in other.pieces.items():
            new_r, new_c = r + shift_r, c + shift_c
            self.pieces[(new_r, new_c)] = pid
            self.id_to_pos[pid] = (new_r, new_c)

def solve_jigsaw_greedy(score_matrix, num_pieces):
    clusters = {pid: JigsawCluster(pid) for pid in range(num_pieces)}
    piece_to_cluster = {pid: pid for pid in range(num_pieces)}
    
    buddies = find_best_buddies(score_matrix)
    pq = []
    for s1, s2, score in buddies:
        heapq.heappush(pq, (-score, s1, s2))
        
    while pq:
        neg_score, s1_idx, s2_idx = heapq.heappop(pq)
        
        p1, side1 = s1_idx // 4, s1_idx % 4
        p2, side2 = s2_idx // 4, s2_idx % 4
        
        c1_id = piece_to_cluster[p1]
        c2_id = piece_to_cluster[p2]
        
        if c1_id == c2_id: continue
        
        cluster1 = clusters[c1_id]
        cluster2 = clusters[c2_id]
        
        r1, c1 = cluster1.get_relative_position(p1)
        r2_local, c2_local = cluster2.get_relative_position(p2)
        
        # Calculate target position of P2 relative to P1
        # 0:Top, 1:Right, 2:Bottom, 3:Left
        dr, dc = 0, 0
        if side1 == 0: # Top matches P2 Bottom -> P2 is Above P1
            dr, dc = -1, 0
        elif side1 == 1: # Right matches P2 Left -> P2 is Right of P1
            dr, dc = 0, 1
        elif side1 == 2: # Bottom matches P2 Top -> P2 is Below P1
            dr, dc = 1, 0
        elif side1 == 3: # Left matches P2 Right -> P2 is Left of P1
            dr, dc = 0, -1
            
        target_r2 = r1 + dr
        target_c2 = c1 + dc
        
        shift_r = target_r2 - r2_local
        shift_c = target_c2 - c2_local
        
        if cluster1.is_compatible(cluster2, shift_r, shift_c):
            cluster1.merge(cluster2, shift_r, shift_c)
            for pid in clusters[c2_id].id_to_pos:
                piece_to_cluster[pid] = c1_id
            del clusters[c2_id]
            
    if not clusters: return {}
    largest = max(clusters.values(), key=lambda c: len(c.pieces))
    return largest.pieces

def reconstruct_image(pieces_dict, pieces_images):
    """
    Reconstructs the image from the pieces dictionary {(r,c): pid}
    """
    if not pieces_dict: return None
    
    rows = [r for r,c in pieces_dict.keys()]
    cols = [c for r,c in pieces_dict.keys()]
    
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Normalize coordinates to 0,0
    h, w, c = pieces_images[0].shape
    
    total_h = (max_r - min_r + 1) * h
    total_w = (max_c - min_c + 1) * w
    
    canvas = np.zeros((total_h, total_w, c), dtype=np.uint8)
    
    for (r, col), pid in pieces_dict.items():
        norm_r = r - min_r
        norm_c = col - min_c
        
        y = norm_r * h
        x = norm_c * w
        
        canvas[y:y+h, x:x+w] = pieces_images[pid]
        
    return canvas
