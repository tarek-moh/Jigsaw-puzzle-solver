import streamlit as st
import cv2
import numpy as np
import solver
import secondary_solver
import os
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide", 
    page_title="Jigsaw Genies",
    page_icon="🧩"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Gradient Background - Deep Professional Blue/Gray */
    .stApp {
        background: linear-gradient(to bottom right, #0F172A, #1E293B);
        color: #F8FAFC;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-enter {
        animation: fadeIn 0.8s ease-out forwards;
    }
    
    /* Custom Title */
    .title-text {
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(120deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s ease-out;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1.2s ease-out;
    }

    /* Cards/Containers */
    .stCard {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Button Styling - Apple AI Inspired Rotating Border (Sandwich Method) */
    .stButton > button {
        position: relative;
        background: transparent !important; /* Transparent to let pseudo-elements show */
        border: none !important;
        color: white !important;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 50px;
        transition: transform 0.2s ease;
        z-index: 1; /* Establish stacking context */
        overflow: visible !important; /* Allow glow to spill */
    }

    /* The Animated Gradient Border/Glow (Bottom Layer) */
    .stButton > button::before {
        content: "";
        position: absolute;
        top: -3px; left: -3px; right: -3px; bottom: -3px;
        border-radius: 53px; /* Slightly larger */
        background: linear-gradient(
            45deg, 
            #00A6FF, #3B82F6, #8B5CF6, #EC4899, #E11D48, #F59E0B
        );
        background-size: 300% 300%;
        animation: gradientBorder 3s linear infinite;
        z-index: -2;
    }

    /* The Inner Dark Body (Middle Layer) */
    .stButton > button::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 50px;
        background: #0F172A; /* Match theme dark blue */
        z-index: -1;
        transition: background 0.3s;
    }
    
    @keyframes gradientBorder {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        /* Outer Glow Effect */
        box-shadow: 
            0 0 15px rgba(59, 130, 246, 0.7), 
            0 0 30px rgba(236, 72, 153, 0.5), 
            0 0 45px rgba(59, 130, 246, 0.3);
    }
    
    /* Slightly lighten inner body on hover */
    .stButton > button:hover::after {
        background: #1E293B;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None

def main():
    # Header
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.write("") 
        st.write("🧩") 
    with col2:
        st.markdown('<p class="title-text">Jigsaw Genies</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle-text">This application demonstrates an automated jigsaw puzzle solver using classical image processing techniques. Select a sample image or upload your own, choose the number of pieces, and watch the AI solve it!</p>', unsafe_allow_html=True)

    st.sidebar.markdown("### ⚙️ Configuration")
    
    input_source = st.sidebar.radio("Select Input Source", ["Sample Images", "Upload Image"])
    
    # Solver Algorithm Selection
    solver_algorithm = st.sidebar.radio("Select Algorithm", ["Cluster-Greedy Assembly (Best-Buddy Constrained)", "Piece-Greedy Sequential Placement"])
    
    num_pieces = st.sidebar.selectbox("Number of Pieces", [4, 16, 64], index=1)
    
    image_path = None
    original_image = None
    
    if input_source == "Sample Images":
        
        folder_map = {
            4: "puzzle_2x2",
            16: "puzzle_4x4",
            64: "puzzle_8x8"
        }
        folder_name = folder_map.get(num_pieces, "correct")
        
        sample_dir = os.path.join("puzzles", "Gravity Falls", folder_name)
        
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files = sorted(files, key=lambda x: int(x.split(".")[0]));
            
            if files:
                selected_file = st.sidebar.selectbox(f"Choose a sample ({num_pieces} pieces)", files)
                image_path = os.path.join(sample_dir, selected_file)
                st.sidebar.success(f"Selected: {selected_file}")
            else:
                st.error(f"No sample images found in 'puzzles/Gravity Falls/{folder_name}'.")
        else:
            st.error(f"Sample directory not found: {sample_dir}")
             
    elif input_source == "Upload Image":
        st.sidebar.info("""
        **⚠️ Upload Requirements:**
        - **Square Images** work best.
        - The image will be **sliced** into the selected number of pieces.
        - Ensure clear, distinct content for best results.
        """)
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_image = cv2.imdecode(file_bytes, 1) # BGR
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    st.divider()
    
    if image_path:
        original_image = load_image(image_path)
        # Store a hash or identifier for the sample image to check for changes
        st.session_state['current_image_id'] = image_path + str(num_pieces)
        
    if original_image is not None:
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("### 🖼️ Original Image")
            st.image(original_image, use_container_width=True, caption="Source Image")
            
        # Check if the image or num_pieces has changed since the last solve
        if 'last_solved_image_id' not in st.session_state or st.session_state['last_solved_image_id'] != st.session_state['current_image_id']:
            if 'solved_image' in st.session_state:
                del st.session_state['solved_image']
            if 'pieces_preview' in st.session_state:
                del st.session_state['pieces_preview']
            if 'solve_metrics' in st.session_state:
                del st.session_state['solve_metrics']
            st.session_state['last_solved_image_id'] = None # Reset solved state

        # Prepare pieces for preview or solving
        if 'pieces_preview' not in st.session_state:
            img_bgr_for_pieces = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            st.session_state['pieces_preview'] = solver.divide_image(img_bgr_for_pieces, num_pieces)
            # Randomize the order for a "scrambled" look
            np.random.shuffle(st.session_state['pieces_preview'])

        with colB:
            if 'solved_image' in st.session_state and st.session_state['solved_image'] is not None:
                st.markdown('<div class="animate-enter">', unsafe_allow_html=True)
                st.markdown("### ✨ Reconstructed Result")
                st.image(st.session_state['solved_image'], use_container_width=True, caption="Solved Puzzle")
                
                # Convert solved image to bytes for download
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(st.session_state['solved_image'], cv2.COLOR_RGB2BGR))
                if is_success:
                    st.download_button(
                        label="Download Solved Image",
                        data=buffer.tobytes(),
                        file_name="solved_jigsaw.png",
                        mime="image/png",
                        use_container_width=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("### 🧩 Scrambled Pieces")
                # Display a grid of scrambled pieces
                grid_size = int(np.sqrt(num_pieces))
                # Ensure grid_size is at least 1 to avoid division by zero or empty columns
                if grid_size == 0: grid_size = 1 
                
                # Create a list of columns for the grid
                cols_preview = st.columns(grid_size)
                
                # Iterate through pieces and display them in the grid
                for idx, piece in enumerate(st.session_state['pieces_preview']):
                    piece_rgb = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
                    with cols_preview[idx % grid_size]:
                        st.image(piece_rgb, use_container_width=True)
                st.caption(f"Deconstructed into {num_pieces} pieces")

        # Preparation for Solving
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            solve_btn = st.button("🧩 SOLVE PUZZLE", use_container_width=True)
            
        if solve_btn:
            st.toast("🧠 AI is analyzing the pieces...")
            with st.spinner("Solving puzzle..."):
                try:
                    import time
                    start_time = time.time()
                    
                    # Use the already divided pieces from session state
                    pieces = st.session_state['pieces_preview']
                    
                    if not pieces:
                        st.error("Failed to divide image.")
                        st.stop()
                        
                    # 2. Extract Strips
                    all_strips = solver.get_all_strips(pieces, border=1)
                    
                    # 3. Calculate Metrics
                    score_matrix = solver.calculate_all_metrics(all_strips)
                    
                    # 4. Solve
                    use_secondary = "Piece-Greedy" in solver_algorithm
                    
                    if use_secondary:
                        # Secondary Solver (Cluster-based)
                        solution_obj = secondary_solver.solve_greedy_newer(score_matrix, num_pieces)
                        # Extract the dictionary format for consistent downstream usage if it's a cluster
                        solution = solution_obj.cluster_array if solution_obj else {}
                    else:
                        # Primary Solver (Greedy)
                        solution = solver.solve_jigsaw_greedy(score_matrix, num_pieces)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    if not solution:
                        st.error("Solver failed to find a consistent solution.")
                    else:
                        # 5. Reconstruct
                        if use_secondary:
                             reconstructed_bgr = solver.reconstruct_image(solution, pieces)
                        else:
                             reconstructed_bgr = solver.reconstruct_image(solution, pieces)
                        
                        if reconstructed_bgr is not None:
                            reconstructed_rgb = cv2.cvtColor(reconstructed_bgr, cv2.COLOR_BGR2RGB)
                            
                            st.session_state['solved_image'] = reconstructed_rgb
                            st.session_state['last_solved_image_id'] = st.session_state['current_image_id']
                            
                            # Store metrics for display
                            st.session_state['solve_metrics'] = {
                                'time': elapsed_time,
                                'score_matrix': score_matrix,
                                'solution': solution
                            }
                            
                            st.balloons()
                            st.success("Puzzle Solved Successfully! 🎉")
                            st.rerun() # Rerun to display the solved image in colB
                            
                            with st.expander("🔍 Inspect Individual Pieces"):
                                st.markdown('<div class="animate-enter">', unsafe_allow_html=True)
                                # Display pieces in a grid
                                grid_size = int(np.sqrt(num_pieces))
                                cols_inspect = st.columns(grid_size)
                                for idx, piece in enumerate(pieces):
                                    piece_rgb = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
                                    with cols_inspect[idx % grid_size]:
                                        st.image(piece_rgb, caption=f"#{idx}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                        else:
                            st.error("Failed to reconstruct image from solution.")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    
        # Show Metrics Dashboard if solved
        if 'solve_metrics' in st.session_state:
            st.divider()
            st.markdown("### 📊 AI Performance Dashboard")
            
            # Retrieve data
            matrix = st.session_state['solve_metrics']['score_matrix']
            time_taken = st.session_state['solve_metrics']['time']
            solution = st.session_state['solve_metrics'].get('solution')
            
            # Compute more intuitive metrics
            # 1. Reconstruct the edges used in the solution to find their confidence scores
            # solution is dict: {(r,c): pid}
            # We need to find neighbors and look up matrix[pid_A_side, pid_B_side]
            
            match_scores = []
            
            # Reverse map to find max R and C
            if solution:
                rows = [k[0] for k in solution.keys()]
                cols = [k[1] for k in solution.keys()]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)
                
                # Iterate through logical grid
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        curr_pid = solution.get((r,c))
                        if curr_pid is None: continue
                        
                        # Check Right Neighbor
                        right_pid = solution.get((r, c+1))
                        if right_pid is not None:
                            # Current Right (1) vs Neighbor Left (3)
                            # Matrix index: pid * 4 + side
                            idx1 = curr_pid * 4 + 1
                            idx2 = right_pid * 4 + 3
                            score = matrix[idx1, idx2]
                            match_scores.append(score)
                            
                        # Check Bottom Neighbor
                        bottom_pid = solution.get((r+1, c))
                        if bottom_pid is not None:
                            # Current Bottom (2) vs Neighbor Top (0)
                            idx1 = curr_pid * 4 + 2
                            idx2 = bottom_pid * 4 + 0
                            score = matrix[idx1, idx2]
                            match_scores.append(score)

            # Analyze Scores
            if match_scores:
                avg_confidence = np.mean(match_scores) * 100
                min_confidence = np.min(match_scores) * 100
                
                # Binning
                high = sum(1 for s in match_scores if s >= 0.8)
                med = sum(1 for s in match_scores if 0.5 <= s < 0.8)
                low = sum(1 for s in match_scores if s < 0.5)
                
                total_edges = len(match_scores)
            else:
                avg_confidence = 0
                match_scores = [0]
                high, med, low = 0, 0, 0

            # --- Display ---
            
            # Row 1: Key KPIs
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric(label="⏱️ Solve Time", value=f"{time_taken:.3f}s")
            with kpi2:
                st.metric(label="🧠 AI Confidence", value=f"{avg_confidence:.1f}%")
            with kpi3:
                st.metric(label="🔗 Total Connections", value=len(match_scores))
                
            st.markdown("#### 📉 Match Quality Distribution")
            st.caption("How sure is the AI about each connection? (High > 80%, Low < 50%)")
            
            # Simple Bar Chart for Distribution
            chart_data = {
                "Quality": ["High Confidence", "Medium Confidence", "Low Confidence"],
                "Count": [high, med, low]
            }
            # Use columns to constrain width if needed, or just st.bar_chart
            # Color by Quality to differentiate bars
            st.bar_chart(chart_data, x="Quality", y="Count", color="Quality")
            
            # Optional: Heatmap in collapsed advanced view for power users
            with st.expander("🔬 View Raw Probability Heatmap (Advanced)"):
                 if matrix is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cax = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
                    fig.colorbar(cax)
                    ax.set_title("Raw Similarity Matrix")
                    ax.axis('off')
                    st.pyplot(fig)

    else:
        st.info("👈 Please select a sample or upload an image from the sidebar to start.")
        # Clear session state if no image is loaded
        if 'solved_image' in st.session_state:
            del st.session_state['solved_image']
        if 'pieces_preview' in st.session_state:
            del st.session_state['pieces_preview']
        if 'current_image_id' in st.session_state:
            del st.session_state['current_image_id']
        if 'last_solved_image_id' in st.session_state:
            del st.session_state['last_solved_image_id']

if __name__ == "__main__":
    main()
