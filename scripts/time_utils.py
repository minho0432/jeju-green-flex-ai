"""오늘 모드의 충전 시작 가능 시간을 계산하는 유틸리티."""

from datetime import datetime


def get_effective_start_hour(
    selected_start_hour: int,
    now: datetime,
    is_today: bool,
) -> int:
    """사용자가 실제로 충전을 시작할 수 있는 가장 이른 정시를 반환한다.

    오늘 모드에서는 현재 시간이 정각이면 현재 시각부터 허용하고,
    1분 이상 지났다면 다음 정시부터 허용한다.
    내일·데모 모드에서는 사용자가 선택한 시작 시각을 그대로 사용한다.
    """
    selected_start_hour = int(selected_start_hour)

    if not is_today:
        return selected_start_hour

    earliest_current_slot = (
        now.hour
        if now.minute == 0
        else now.hour + 1
    )

    return max(selected_start_hour, earliest_current_slot)