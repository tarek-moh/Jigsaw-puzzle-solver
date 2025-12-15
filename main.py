from batch_solver import run_batch_puzzle_test


def main():
    # 1. Configuration for Single Run
    # You can change this to run a specific single puzzle and see the result
    image_path = r"puzzles\Gravity Falls\puzzle_4x4\22.jpg"
    num_pieces = 16

    # Run Single Puzzle
    # solve_puzzle(image_path, num_pieces, output_path="reconstructed_result.png", show_result=True)

    # Uncomment the following lines to run batch processing instead:
    # source_dir = r"puzzles\Gravity Falls\puzzle_8x8"
    # output_dir = r"results\puzzle_8x8"
    # run_batch_puzzle_test(source_dir, output_dir, num_pieces=64)

    # Uncomment the following lines to run the secondary batch processing instead:
    source_dir = r"puzzles\Gravity Falls\puzzle_8x8"
    output_dir = r"secondary_results\puzzle_8x8"
    run_batch_puzzle_test(source_dir, output_dir, num_pieces=64, use_secondary_function=True)

if __name__ == "__main__":
    main()
