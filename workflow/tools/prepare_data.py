# -*- coding: utf-8 -*-
"""
장비 정비이력 엑셀 → Orange3 네이티브 테이블(.tab) 변환기

실습지(데이터분석_실습지.md) '0. 환경 설정' + '문제 1-1' 에 해당한다.
파이썬에서는 pd.read_excel(..., sheet_name='데이터') 한 줄이지만,
Orange 는 컬럼의 '역할(타깃/메타/피처)'과 '타입(범주형/수치형/시간)'을
파일 헤더에 미리 적어두면 이후 모든 워크플로우가 설정 없이 바로 동작한다.

생성물
  workflow/data/equipment_maintenance.tab  : 원본 5,000건 (결측 그대로 유지)
  workflow/data/equipment_new_input.tab    : 예측 앱 입력용 5건 (타깃 = ?)

실행
  /Applications/Orange.app/Contents/MacOS/python workflow/tools/prepare_data.py
  (또는 pandas·openpyxl 이 설치된 아무 파이썬)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # my-eda-app/
OUT = Path(__file__).resolve().parents[1] / "data"  # workflow/data/
SRC = ROOT / "data" / "dataset1_equipment_maintenance.xlsx"
SHEET = "데이터"
TARGET = "failure_within_30days"

# 컬럼명 -> (Orange 타입, 역할)  · 역할 "" 이면 일반 피처
SCHEMA = [
    ("equipment_id",                "string",     "meta"),   # 그룹 분할 기준 (모델 피처 아님)
    ("equipment_type",              "discrete",   ""),
    ("unit_code",                   "discrete",   ""),
    ("equipment_age_years",         "continuous", ""),
    ("maintenance_date",            "time",       "meta"),   # 시계열 축 (모델 피처 아님)
    ("maintenance_type",            "discrete",   ""),
    ("fault_code",                  "discrete",   ""),       # 결측 2008건 (예방정비)
    ("fault_category",              "discrete",   ""),       # 결측 2008건 (예방정비)
    ("severity",                    "continuous", ""),       # 0~5 서열 -> 상관분석 위해 수치형
    ("downtime_hours",              "continuous", ""),
    ("repair_duration_hours",       "continuous", ""),
    ("parts_replaced_count",        "continuous", ""),
    ("maintenance_cost",            "continuous", ""),
    ("maintenance_success",         "discrete",   ""),
    ("cumulative_operating_hours",  "continuous", ""),
    ("days_since_last_maintenance", "continuous", ""),       # 결측 200건 (최초 정비)
    ("availability_rate",           "continuous", ""),
    (TARGET,                        "discrete",   "class"),  # 타깃: 30일 내 재고장
    ("group_fold",                  "discrete",   "meta"),   # 장비 단위 5-폴드 (GroupKFold 대응)
]


def add_group_fold(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """장비(equipment_id) 단위로 1~5 폴드 번호를 부여한다.

    파이썬 실습지의 GroupKFold(n_splits=5) 를 Orange 에서 재현하기 위한 컬럼.
    Test and Score 위젯의 'Cross validation by feature' 에 이 컬럼을 지정하면
    같은 장비가 학습·검증 폴드에 동시에 들어가지 않는다.
    """
    equipment = sorted(df["equipment_id"].unique())
    import numpy as np
    order = np.random.RandomState(42).permutation(len(equipment))
    fold_of = {eq: str(order[i] % n_folds + 1) for i, eq in enumerate(equipment)}
    df["group_fold"] = df["equipment_id"].map(fold_of)
    return df


def to_tab(df: pd.DataFrame, path: Path) -> None:
    """3줄 헤더(이름/타입/역할)를 가진 Orange 네이티브 .tab 파일로 저장."""
    cols = [c for c, _, _ in SCHEMA]
    lines = [
        "\t".join(cols),
        "\t".join(t for _, t, _ in SCHEMA),
        "\t".join(r for _, _, r in SCHEMA),
    ]
    for row in df[cols].itertuples(index=False, name=None):
        cells = []
        for value in row:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("?")          # Orange 의 결측 표기
            else:
                cells.append(str(value))
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장 완료 → {path}  ({len(df):,}행 x {len(cols)}열)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SRC, sheet_name=SHEET)
    df["maintenance_date"] = pd.to_datetime(df["maintenance_date"]).dt.strftime("%Y-%m-%d")
    print("shape :", df.shape)                       # 실습지 기대 출력: (5000, 18)
    print("재고장 비율 :", round(df[TARGET].mean() * 100, 1), "%")   # 기대: 19.5 %
    print("결측 :", df.isnull().sum()[lambda s: s > 0].to_dict())
    df = add_group_fold(df)

    to_tab(df, OUT / "equipment_maintenance.tab")

    # 예측 앱(실습 4)용 입력 샘플 — 타깃을 지워 '아직 모르는 신규 건'으로 만든다
    sample = df.sample(5, random_state=42).copy()
    sample[TARGET] = None
    to_tab(sample, OUT / "equipment_new_input.tab")


if __name__ == "__main__":
    main()
