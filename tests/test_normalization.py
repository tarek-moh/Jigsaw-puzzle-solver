import numpy as np
import solver

def test_normalization():
    print("Testing normalization functions...")
    
    # Test SSD Normalization
    ssd_input = np.array([0, 50, 100])
    ssd_expected = np.array([1.0, 0.5, 0.0])
    ssd_output = solver.normalize_ssd(ssd_input)
    assert np.allclose(ssd_output, ssd_expected), f"SSD Failed: {ssd_output}"
    print("SSD Normalization: PASS")
    
    # Test Correlation Normalization
    corr_input = np.array([-1.0, 0.0, 1.0])
    corr_expected = np.array([0.0, 0.5, 1.0])
    corr_output = solver.normalize_correlation(corr_input)
    assert np.allclose(corr_output, corr_expected), f"Correlation Failed: {corr_output}"
    print("Correlation Normalization: PASS")
    
    # Test Gradient Normalization (Same as SSD)
    grad_input = np.array([0, 10, 20])
    grad_expected = np.array([1.0, 0.5, 0.0])
    grad_output = solver.normalize_gradient(grad_input)
    assert np.allclose(grad_output, grad_expected), f"Gradient Failed: {grad_output}"
    print("Gradient Normalization: PASS")
    
    # Test Combined Score
    metrics = {
        'ssd': np.array([1.0, 0.0]),
        'corr': np.array([1.0, 0.0])
    }
    weights = {'ssd': 0.5, 'corr': 0.5}
    combined_expected = np.array([1.0, 0.0])
    combined_output = solver.calculate_composite_score(metrics, weights)
    assert np.allclose(combined_output, combined_expected), f"Combined Failed: {combined_output}"
    print("Combined Score: PASS")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_normalization()
