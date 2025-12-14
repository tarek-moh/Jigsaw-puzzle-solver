import cv2
import numpy as np
import matplotlib.pyplot as plt
import solver
import os

def main():
    # 1. Configuration
    image_path = r"puzzles\Gravity Falls\puzzle_4x4\0.jpg"
    num_pieces = 16 # 4x4 grid
    
    # 2. Load Image
    print(f"Loading image from: {image_path}")
    original_image = solver.readImage(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return

    # 3. Divide into pieces
    print(f"Dividing image into {num_pieces} pieces...")
    pieces = solver.divide_image(original_image, num_pieces)

    # 4. Extract Strips
    print("Extracting strips...")
    all_strips = solver.get_all_strips(pieces)
    
    # 5. Calculate Metrics & Composite Score
    print("Calculating metrics and scores...")
    score_matrix = solver.calculate_all_metrics(all_strips)
    
    # 6. Solve
    print("Solving puzzle...")
    # The solver returns a dict: {(row, col): piece_id}
    solution = solver.solve_jigsaw_greedy(score_matrix, num_pieces)
    
    if not solution:
        print("Solver failed to return a solution.")
        return
        
    print("Solution found!")
    print(solution)
    
    # 7. Reconstruct
    print("Reconstructing image...")
    reconstructed_img = solver.reconstruct_image(solution, pieces)
    
    if reconstructed_img is not None:
        # Show Original vs Reconstructed
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Reconstructed")
        plt.imshow(cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save result
        output_path = "reconstructed_result.png"
        cv2.imwrite(output_path, reconstructed_img)
        print(f"Result saved to {output_path}")
    else:
        print("Failed to reconstruct image.")

if __name__ == "__main__":
    main()
