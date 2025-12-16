from batch_solver import run_batch_puzzle_test


def main():
    # 1. Configuration
    # Options: 'PRIMARY', 'SECONDARY', 'BOTH'
    SOLVER_TYPE = 'BOTH'
    
    # Parameters
    image_path = r"puzzles\Gravity Falls\puzzle_4x4\22.jpg"
    num_pieces = 16
    
    # Batch Processing Parameters
    run_batch = False
    source_dir = r"puzzles\Gravity Falls\puzzle_2x2"
    output_dir_primary = r"results\puzzle_2x2"
    output_dir_secondary = r"secondary_results\puzzle_2x2"
    batch_num_pieces = 4

    # ---------------------------------------------------------
    
    # 2. Run Single Puzzle
    if not run_batch:
        from batch_solver import solve_puzzle
        print(f"--- Running Single Puzzle ({SOLVER_TYPE}) ---")
        
        if SOLVER_TYPE in ['PRIMARY', 'BOTH']:
            print("\n[Primary Solver]")
            solve_puzzle(image_path, num_pieces, output_path="reconstructed_result_primary.png", show_result=True, use_secondary_function=False)
            
        if SOLVER_TYPE in ['SECONDARY', 'BOTH']:
            print("\n[Secondary Solver]")
            solve_puzzle(image_path, num_pieces, output_path="reconstructed_result_secondary.png", show_result=True, use_secondary_function=True)

    # 3. Run Batch Processing
    else:
        print(f"--- Running Batch Processing ({SOLVER_TYPE}) ---")
        
        if SOLVER_TYPE in ['PRIMARY', 'BOTH']:
            print("\n[Primary Solver Batch]")
            run_batch_puzzle_test(source_dir, output_dir_primary, num_pieces=batch_num_pieces, use_secondary_function=False)
            
        if SOLVER_TYPE in ['SECONDARY', 'BOTH']:
            print("\n[Secondary Solver Batch]")
            run_batch_puzzle_test(source_dir, output_dir_secondary, num_pieces=batch_num_pieces, use_secondary_function=True)

if __name__ == "__main__":
    main()
