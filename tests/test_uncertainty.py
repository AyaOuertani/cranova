import numpy as np
from uncertainty import conformal_quantile, interval_half_widths


def test_quantile_covers_the_requested_share():
    residuals = np.arange(1, 20, dtype=float)[:, None]
    assert conformal_quantile(residuals, 0.9)[0] == 18
    
def test_quantile_never_exceeds_the_largest_error():
    residuals = np.arange(1, 10, dtype=float)[:, None]
    assert conformal_quantile(residuals, 0.99)[0] == 9
    
def test_group_intervals_use_the_group_own_errors():
    # group a: small errors, group b: large
    residuals = np.vstack([np.full((30, 2), 1.0), np.full((30, 2), 5.0)])
    train_groups = np.array(["a"] * 30 + ["b"] * 30)
    half = interval_half_widths(residuals, train_groups, np.array(["a", "b"]), 0.9)
    assert half["Per distance + group"][0, 0] == 1.0 and half["Per distance + group"][1, 0] == 5.0
    assert half["Per distance"][0, 0] == half["Per distance"][1,0]    