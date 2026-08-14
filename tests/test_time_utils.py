from datetime import datetime

from scripts.time_utils import get_effective_start_hour


def test_today_exact_hour_allows_current_hour():
    now = datetime(2026, 8, 14, 16, 0)

    result = get_effective_start_hour(
        selected_start_hour=8,
        now=now,
        is_today=True,
    )

    assert result == 16


def test_today_after_exact_hour_moves_to_next_hour():
    now = datetime(2026, 8, 14, 16, 1)

    result = get_effective_start_hour(
        selected_start_hour=8,
        now=now,
        is_today=True,
    )

    assert result == 17


def test_today_preserves_later_user_selected_hour():
    now = datetime(2026, 8, 14, 16, 20)

    result = get_effective_start_hour(
        selected_start_hour=20,
        now=now,
        is_today=True,
    )

    assert result == 20


def test_today_after_23_hour_returns_24():
    now = datetime(2026, 8, 14, 23, 1)

    result = get_effective_start_hour(
        selected_start_hour=8,
        now=now,
        is_today=True,
    )

    assert result == 24


def test_tomorrow_does_not_apply_current_time_filter():
    now = datetime(2026, 8, 14, 16, 20)

    result = get_effective_start_hour(
        selected_start_hour=8,
        now=now,
        is_today=False,
    )

    assert result == 8