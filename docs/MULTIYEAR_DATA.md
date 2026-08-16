# 다년도 학습 데이터 근거

## 최종 병합 결과

- 기간: 2023-01-01 00:00 ~ 2025-12-31 23:00
- 총 25,559시간: 2023년 8,015 + 2024년 8,784 + 2025년 8,760
- 기존 2025년 데이터보다 16,799시간 추가
- 제주 계통수요는 모든 행에 공식 관측값을 연결하고 MWh로 통일
- SMP는 값이 있는 2025년 8,760시간만 보조 모델에 사용하며 Green Score에는 사용하지 않음

## 출처와 처리

| 자료 | 출처 | 처리 |
|---|---|---|
| 지역별 태양광·풍력 | KPX 공공데이터포털 파일자료 | 제주 행 선택, 시간별 MWh로 정리 |
| 제주 계통수요 | KPX 공공데이터포털 파일자료 | 2023년 구간의 kWh 값을 MWh로 변환 후 결합 |
| 기온·습도·풍속·일사량 | Open-Meteo Historical Weather API | Asia/Seoul 시간별 자료로 결합 |

원본 링크:

- https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15065269
- https://www.data.go.kr/data/15065239/fileData.do?recommendDataYn=Y
- https://open-meteo.com/en/docs/historical-weather-api

## 결측 처리 원칙

- 2023년 12월 발전 원본은 같은 공개자료에서 확보되지 않아 임의 생성하지 않음
- 2023-01-26 00:00 풍력값 1건은 원본이 비어 있어 해당 시간 전체를 제외
- 연도 경계나 누락 구간을 가로지르는 과거 24·48·168시간 특징은 만들지 않음
- 최종 병합은 `scripts/build_multiyear_data.py`, 검사는 `scripts/validate_data.py`로 재현
