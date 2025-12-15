import cv2
import numpy as np
import matplotlib.pyplot as plt
import solver
import os

def solve_puzzle(image_path, num_pieces, output_path=None, show_result=False):
    """
    Solves a single puzzle.
    If output_path is provided, saves the result.
    If show_result is True, displays the plots.
    Returns True if successful, False otherwise.
    """
    # 2. Load Image
    print(f"Loading image from: {image_path}")
    original_image = solver.readImage(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return False

    # 3. Divide into pieces
    print(f"Dividing image into {num_pieces} pieces...")
    pieces = solver.divide_image(original_image, num_pieces)

    # 4. Extract Strips
    print("Extracting strips...")
    all_strips = solver.get_all_strips(pieces, border=1)

    # 5. Calculate Metrics & Composite Score
    print("Calculating metrics and scores...")
    score_matrix = solver.calculate_all_metrics(all_strips)

    # 6. Solve
    print("Solving puzzle...")
    # The solver returns a dict: {(row, col): piece_id}
    solution = solver.solve_jigsaw_greedy(score_matrix, num_pieces)

    if not solution:
        print("Solver failed to return a solution.")
        return False

    print("Solution found!")
    # print(solution)

    # 7. Reconstruct
    print("Reconstructing image...")
    reconstructed_img = solver.reconstruct_image(solution, pieces)

    if reconstructed_img is not None:
        if output_path:
            cv2.imwrite(output_path, reconstructed_img)
            print(f"Result saved to {output_path}")

        if show_result:
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

        return True
    else:
        print("Failed to reconstruct image.")
        return False

def run_batch_puzzle_test(source_folder, output_folder, num_pieces):
    """
    Iterates over all images in source_folder, solves them, and saves results to output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    files.sort()

    print(f"Found {len(files)} images in {source_folder}")

    success_count = 0
    for filename in files:
        full_path = os.path.join(source_folder, filename)
        save_path = os.path.join(output_folder, f"result_{filename}")

        print(f"\nProcessing {filename}...")
        if solve_puzzle(full_path, num_pieces, output_path=save_path, show_result=False):
            success_count += 1

    print(f"\nBatch processing complete. {success_count}/{len(files)} puzzles solved successfully.")

def main():
    # 1. Configuration for Single Run
    # You can change this to run a specific single puzzle and see the result
    image_path = r"puzzles\Gravity Falls\puzzle_4x4\22.jpg"
    num_pieces = 16

    # Run Single Puzzle
    # solve_puzzle(image_path, num_pieces, output_path="reconstructed_result.png", show_result=True)

    # Uncomment the following lines to run batch processing instead:
    source_dir = r"puzzles\Gravity Falls\puzzle_8x8"
    output_dir = r"results\puzzle_8x8"
    run_batch_puzzle_test(source_dir, output_dir, num_pieces=64)

if __name__ == "__main__":
    main()
