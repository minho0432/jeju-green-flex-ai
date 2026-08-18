def calculate_green_point(schedule, base_rate=100.0):
    eligible = schedule[schedule["score"] >= 70].copy()

    if eligible.empty:
        return {
            "eligible_kwh": 0.0,
            "weighted_score": 0.0,
            "expected_points": 0,
        }

    energy = eligible["scheduled_kwh"].sum()

    weighted_score = (
        (eligible["scheduled_kwh"] * eligible["score"]).sum()
        / energy
    )

    points = energy * base_rate * (weighted_score / 100.0)

    return {
        "eligible_kwh": float(energy),
        "weighted_score": float(weighted_score),
        "expected_points": round(points),
    }