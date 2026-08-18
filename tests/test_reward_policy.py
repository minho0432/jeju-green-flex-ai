import pandas as pd

from scripts.reward_policy import calculate_green_point


def test_green_point_example():
    schedule = pd.DataFrame({
        "scheduled_kwh": [40],
        "score": [85],
    })

    result = calculate_green_point(schedule, base_rate=100)

    assert result == {
        "eligible_kwh": 40.0,
        "weighted_score": 85.0,
        "expected_points": 3400,
    }