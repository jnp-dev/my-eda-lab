# 🍊 Orange3 실습 워크플로우

`데이터분석_실습지.md` 의 실습 순서를 Orange3 에서 코드 없이 따라갈 수 있도록 옮긴
워크플로우 모음이다. 파일 번호가 곧 실습 순서이므로 **01 → 08 순서대로** 열면 된다.

---

## 시작하기

1. Orange3 를 실행한다. (macOS: `/Applications/Orange.app`, Windows: Orange3 바로가기)
2. `File ▸ Open` 에서 `workflow/01_eda.ows` 를 연다.
3. 캔버스의 **회색 글상자(주석)** 가 실습지의 문제 번호와 설정 방법을 알려준다.
   위젯을 더블클릭해 주석대로 설정하고, 결과가 나오면 다음 위젯으로 넘어간다.

> 데이터 경로는 워크플로우 파일 기준 상대경로로 저장돼 있다.
> `workflow/` 폴더를 통째로 옮겨도 그대로 열린다.

### 데이터 파일

`data/equipment_maintenance.tab` 은 원본 엑셀(`data/dataset1_equipment_maintenance.xlsx`
의 `데이터` 시트, 5,000행)을 Orange 네이티브 형식으로 바꾼 것이다.
헤더에 타입과 역할이 적혀 있어 워크플로우를 열면 **처음부터 아래 상태**로 잡혀 있다.

| 역할 | 컬럼 |
| --- | --- |
| 타깃 (class) | `failure_within_30days` |
| 메타 | `equipment_id`, `maintenance_date`, `group_fold` |
| 피처 | 나머지 15개 |

`group_fold` 는 원본에 없는 추가 컬럼이다. **장비(equipment_id) 단위로 부여한 5-폴드 번호**로,
Test and Score 의 `Cross validation by feature` 에 지정하면 실습지의 `GroupKFold(n_splits=5)` 가 된다.

엑셀을 직접 바꿨다면 다시 만들면 된다.

```bash
/Applications/Orange.app/Contents/MacOS/python workflow/tools/prepare_data.py
```

---

## 워크플로우 ↔ 실습지 대응표

| 파일 | 실습지 | 내용 |
| --- | --- | --- |
| `01_eda.ows` | 실습 1 (1-1 ~ 1-6) | 데이터 확인 · 결측 · 타깃 불균형 · 시각화 5종 · 결측 처리 · IQR 이상치 |
| `02_dashboard.ows` | 실습 2 (2-1) | 필터 UI · KPI · 유형별 재고장률 · 비용 분포 · 월별 추이 |
| `03_split.ows` | 실습 3-1 ⭐, 3-6 | 그룹 분할(중복 장비 0) · 무작위 분할 함정 비교 · 정보성 결측 확인 |
| `04_model_compare.ows` | 실습 3-2 | Dummy 베이스라인 + 7개 모델 · 불균형 보정 · 혼동행렬/ROC |
| `05_tuning_threshold.ows` | 실습 3-3, 3-4 ⭐ | 파라미터 튜닝 · 임계값 최적화 · 목표 Recall 0.80 |
| `06_interpretation.ows` | 실습 3-5 | 피처 중요도 · 예측 근거(Nomogram) · 트리 규칙 |
| `07_predict_app.ows` | 실습 4 (4-1 ~ 4-3) | 모델 저장/로드 · 입력 폼 예측 · predictions.csv 기록 |
| `08_lab_visualization.ows` | 선택 Lab A | 투영 시각화 · t-SNE · PCA · 군집 · Heat Map |

### 파이썬 코드 ↔ Orange 위젯 대응

| 실습지 파이썬 | Orange 위젯 |
| --- | --- |
| `pd.read_excel(...)` | **File** |
| `df.info()` / `df.describe()` | **Data Info** · **Column Statistics** |
| `df.isnull().sum()` | **Column Statistics** 의 Missing 열 |
| `value_counts().plot.bar()` | **Distributions** |
| `sns.boxplot` / `px.box` | **Box Plot** · **Violin Plot** |
| `plt.scatter` | **Scatter Plot** |
| 상관행렬 히트맵 | **Correlations** (+ Scatter Plot 연동) |
| `fillna` / `is_first_maintenance` | **Impute** (As a distinct value) + **Purge Domain** |
| `st.multiselect` / `st.slider` | **Select Rows** |
| `st.metric` KPI | **Data Info** · **Pivot Table** |
| `df.groupby().mean()` | **Group by** · **Pivot Table** |
| `GroupShuffleSplit` | **Python Script** (03 워크플로우) |
| `GroupKFold(n_splits=5)` | **Test and Score** ▸ Cross validation by feature ▸ `group_fold` |
| `DummyClassifier` | **Constant** |
| `class_weight='balanced'` | 학습 위젯의 *Balance class distribution* 체크박스 |
| `cross_validate` / `classification_report` | **Test and Score** · **Confusion Matrix** |
| Optuna 튜닝 | **Parameter Fitter** (+ 학습 위젯 파라미터 직접 조정) |
| 임계값 튜닝 | **Calibrated Learner** · **ROC Analysis** · Python Script |
| SHAP summary / waterfall | **Rank** · **Nomogram** (Explain 애드온 설치 시 진짜 SHAP) |
| `joblib.dump/load` | **Save Model** · **Load Model** |
| `predict_app.py` 입력 폼 | **Create Instance** |
| `predictions.csv` 누적 | **Save Data** |

---

## Orange 로는 그대로 안 되는 것 (그리고 이 워크플로우의 대응)

정직하게 짚고 넘어갈 부분이다. 실습지의 몇 가지는 Orange 에 대응 위젯이 없다.

| 실습지 항목 | 상황 | 이 워크플로우의 처리 |
| --- | --- | --- |
| `GroupShuffleSplit` | Data Sampler 에 그룹 분할 없음 | **Python Script** 위젯으로 구현 (03). 코드가 다 적혀 있으니 Run 만 누르면 된다 |
| PR-AUC (`average_precision`) | Test and Score 에 지표 없음 | AUC + Recall + Precision 으로 대체. PR-AUC 숫자는 05 의 Python Script 가 출력 |
| Optuna 베이지안 최적화 | 자동 탐색기 없음 | **Parameter Fitter** 로 파라미터별 과적합 지점을 눈으로 찾는다 |
| 목표 Recall 임계값 | ROC Analysis 는 비용비 기반 | 05 의 Python Script 가 Recall ≥ 0.80 임계값을 직접 계산 |
| SHAP | 기본 배포에 없음 | Rank + Nomogram 으로 대체. 아래 애드온 설치 시 진짜 SHAP 사용 가능 |
| ydata-profiling / sweetviz | 대응 없음 | Column Statistics + Distributions 조합이 가장 가깝다 |
| MLflow (Lab B) | 대응 없음 | Orange 는 워크플로우 파일 자체가 실험 기록이다. 조건별로 `.ows` 를 저장해 비교 |
| Prefect (Lab C) | 대응 없음 | 스케줄 실행이 필요하면 파이썬 `ml_pipeline.py` 를 쓴다 |
| 파이 차트 | 대응 없음 | Distributions(막대)로 본다 |

### SHAP 을 쓰고 싶다면 — Explain 애드온

```
Orange 메뉴 ▸ Options ▸ Add-ons... ▸ "Explain" 체크 ▸ OK ▸ Orange 재시작
```

설치하면 위젯 목록에 **Explain Model**(beeswarm/bar = SHAP summary)과
**Explain Prediction**(SHAP waterfall)이 생긴다.
`06_interpretation.ows` 를 열고 Train 스크립트와 학습 위젯의 Model 출력을 이 두 위젯에
연결하면 실습지 문제 3-5 를 그대로 재현할 수 있다.

---

## 실행 순서 요령

- **03 → 08 워크플로우에는 Python Script 위젯이 있다.** 위젯을 열고 **Run(⌘R / Ctrl+R)** 을
  한 번 눌러야 아래로 데이터가 흐른다. 창을 닫아도 결과는 유지된다.
- **07 은 두 단계다.** 위쪽 줄(학습 → Save Model)을 먼저 실행해
  `artifacts/model.pkcls` 를 만든 뒤에야 아래쪽 Load Model 이 파일을 찾는다.
- **Test and Score 에서 Target class 를 `1` 로 바꾼다.** 그래야 Recall/Precision/F1 이
  '재고장' 기준이 된다. 기본값(평균)으로 두면 불균형 데이터에서 숫자가 낙관적으로 보인다.
- 08 의 t-SNE 는 느리므로 Data Sampler(20%) 를 거치도록 해 두었다.

## 스스로 확인할 체크리스트

실습지의 제출 체크리스트를 Orange 기준으로 옮긴 것이다.

- [ ] 03 · '중복 장비 검증' 스크립트가 **중복 장비 수 = 0** 을 출력하는가
- [ ] 04 · **Constant(Dummy)** 를 포함해 비교했고, Dummy 의 CA 가 왜 높은지 설명할 수 있는가
- [ ] 04 · Target class 를 `1` 로 두고 **Recall** 을 봤는가
- [ ] 05 · 임계값을 낮췄을 때 **FN 이 얼마나 줄고 FP 가 얼마나 늘었는지** 숫자로 말할 수 있는가
- [ ] 03 · 결측을 최빈값으로 채우면 왜 "틀린 처리"인지 설명할 수 있는가
- [ ] 07 · Create Instance 의 초기값을 **Median** 으로 두었는가
- [ ] 각 워크플로우 주석의 **생각해보기** 질문에 답했는가

---

## 폴더 구조

```
workflow/
├── README.md                     이 문서
├── 01_eda.ows ~ 08_....ows       실습 워크플로우 (번호 순서대로 진행)
├── data/
│   ├── equipment_maintenance.tab 실습용 데이터 (5,000행, 역할/타입 지정 완료)
│   └── equipment_new_input.tab   예측 앱 입력 샘플 5건 (타깃 비어 있음)
├── artifacts/                    07 에서 저장할 모델(model.pkcls) 위치
├── reports/                      07 에서 저장할 predictions.csv 위치
└── tools/
    ├── prepare_data.py           엑셀 → .tab 변환
    ├── build_workflows.py        .ows 생성기 (워크플로우를 고칠 때 사용)
    └── validate_workflows.py     생성된 .ows 를 Orange 로더로 검증
```

`tools/` 는 실습에 필요하지 않다. 워크플로우 내용을 수정하려면
`build_workflows.py` 를 고친 뒤 아래를 실행한다.

```bash
/Applications/Orange.app/Contents/MacOS/python workflow/tools/build_workflows.py
/Applications/Orange.app/Contents/MacOS/python workflow/tools/validate_workflows.py
```
