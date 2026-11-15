"""
데이터셋 프로파일 — dataset_usage_guide.docx 기반 참조 메타데이터.

목적
- 알려진 교육용 데이터셋(DS1/DS2/DS3)이면 컬럼 설명·추천 타깃·추천 기법·해석 포인트를
  자동으로 제공한다.
- 모르는 데이터셋(일반 CSV/XLSX)이면 detect_profile()가 None 을 반환하고,
  앱은 컬럼 타입만 보고 동작하는 '범용 분석 모드'로 작동한다.

즉 이 파일은 '특정 데이터에 대한 친절한 설명'일 뿐, 앱 동작의 전제가 아니다.
"""

# 일반 데이터에서 타깃 컬럼으로 흔히 쓰이는 이름들 (소문자 비교)
COMMON_TARGET_NAMES = [
    'target', 'label', 'y', 'class', 'outcome', 'result',
    'is_anomaly', 'anomaly', '타겟', '레이블', '정답',
]

PROFILES = [
    {
        'key': 'DS1',
        'name': '장비 정비이력 및 가동률',
        'objective': '예측정비(Predictive Maintenance) — 장비의 30일 내 재고장 여부 예측',
        'category': '분류 / 회귀',
        'signature': ['failure_within_30days', 'maintenance_type', 'severity'],
        'target': 'failure_within_30days',
        'task': 'classification',
        'techniques': ['이진분류(RandomForest/XGBoost)', '회귀(downtime_hours/maintenance_cost)',
                       '생존분석(RUL)', 'Data Leakage 실습', 'XAI(SHAP)'],
        'leakage': ['cumulative_operating_hours', 'availability_rate'],
        'eda_notes': [
            '긴급정비의 downtime_hours가 현저히 높음 → 예방정비 비중 확대 근거',
            'equipment_age_years ↑ → failure_within_30days=1 비율 증가 → 노후 장비 교체 기준',
            'severity ↔ maintenance_cost 강한 양의 상관 → 조기 탐지의 비용 절감 효과',
            'N/A(예방정비 건) 처리: 제거 vs 별도 인코딩 → 결측 전략 학습',
        ],
        'columns': {
            'equipment_id': ('장비 고유번호', 'VARCHAR', 'EQ-0001~EQ-0200'),
            'equipment_type': ('장비 유형', 'VARCHAR', '차량/통신장비/발전기/화기'),
            'unit_code': ('소속 부대코드', 'VARCHAR', 'UNIT-01~UNIT-20'),
            'equipment_age_years': ('장비 도입 후 경과연수', 'FLOAT', '잔존수명(RUL) 학습 활용'),
            'maintenance_date': ('정비 일자', 'DATE', '2023-01-01~2025-12-31'),
            'maintenance_type': ('정비 구분', 'VARCHAR', '예방정비/사후정비/긴급정비'),
            'fault_code': ('고장 코드', 'VARCHAR', 'F001~F050, 예방정비 시 N/A'),
            'fault_category': ('고장 유형', 'VARCHAR', '엔진/전기/유압/통신/기타, 예방정비 시 N/A'),
            'severity': ('고장 심각도', 'INT', '0=해당없음 ~ 5=치명'),
            'downtime_hours': ('가동중단 시간(h)', 'FLOAT', ''),
            'repair_duration_hours': ('수리 소요시간(h)', 'FLOAT', ''),
            'parts_replaced_count': ('교체 부품 수', 'INT', ''),
            'maintenance_cost': ('정비 비용', 'INT', '만원 단위'),
            'maintenance_success': ('정비 성공 여부', 'INT', '0=실패/재정비, 1=성공'),
            'cumulative_operating_hours': ('누적 가동시간(h)', 'FLOAT', '⚠ Data Leakage 주의'),
            'days_since_last_maintenance': ('직전 정비 후 경과일', 'INT', '최초 정비 건은 N/A'),
            'availability_rate': ('해당 월 장비 가동률', 'FLOAT', '0.0~1.0, ⚠ Data Leakage 주의'),
            'failure_within_30days': ('30일 내 재고장 여부', 'INT', '0/1, ★타겟변수★'),
        },
    },
    {
        'key': 'DS2',
        'name': '기상 관측소별 시계열',
        'objective': '시계열 예측(Time Series Forecasting) — 관측소별 미래 기온 예측, 계절성·주기성 분석',
        'category': '시계열 예측 / 다중분류',
        'signature': ['temperature', 'precipitation', 'station_region'],
        'target': 'temperature',
        'task': 'regression',
        'techniques': ['ARIMA/SARIMA', 'Prophet', 'LSTM/GRU', '주기성 인코딩(sin/cos)',
                       '다중분류(weather_alert_level)', '전이학습(지역 간)'],
        'leakage': ['feels_like_temp'],   # 기온 기반 파생값 → 기온 예측 시 누수
        'eda_notes': [
            'ACF 24시간 주기 피크 → 일변동 계절성, SARIMA seasonal order 근거',
            '해안/산악 지역 풍속 차이 → 지역별 모델 분리/전이학습 근거',
            '강수량 0값 편중(약 84%) → 불균형 대응 필수(SMOTE, threshold)',
            '기온-습도 역상관 → 다변량 입력 시 multicollinearity 주의',
        ],
        'columns': {
            'station_id': ('관측소 ID', 'VARCHAR', 'WS-01~WS-10'),
            'station_region': ('관측소 지역', 'VARCHAR', '전방/후방/해안/산악/내륙'),
            'observation_datetime': ('관측 일시', 'DATETIME', '1시간 단위'),
            'hour': ('시간', 'INT', '0~23, 피처 엔지니어링용'),
            'day_of_week': ('요일', 'INT', '0(월)~6(일)'),
            'month': ('월', 'INT', '1~12, 계절성 학습용'),
            'temperature': ('기온(℃)', 'FLOAT', '★시계열 예측 타겟(t+1/t+3)★'),
            'humidity': ('습도(%)', 'FLOAT', '0~100'),
            'wind_speed': ('풍속(m/s)', 'FLOAT', ''),
            'wind_direction': ('풍향(°)', 'INT', '0~360'),
            'atmospheric_pressure': ('기압(hPa)', 'FLOAT', ''),
            'precipitation': ('강수량(mm)', 'FLOAT', '대부분 0 → 불균형 교육'),
            'visibility': ('시정거리(km)', 'FLOAT', ''),
            'cloud_cover': ('운량', 'INT', '0~10 (기상청 기준)'),
            'feels_like_temp': ('체감온도(℃)', 'FLOAT', '기온·풍속·습도 파생값'),
            'weather_alert_level': ('기상특보 등급', 'INT', '0없음/1주의보/2경보/3특보'),
        },
    },
    {
        'key': 'DS3',
        'name': '국방망 트래픽 부하 및 장애이력',
        'objective': '이상탐지(Anomaly Detection) + 분류 — 트래픽 이상 패턴 탐지 및 장애 유형 분류',
        'category': '이상탐지 / 이진·다중분류',
        'signature': ['is_anomaly', 'failure_type', 'connections_per_ip'],
        'target': 'is_anomaly',
        'task': 'classification',
        'techniques': ['지도분류(RF/SVM)', '다중분류(failure_type 4-class)',
                       '비지도 이상탐지(IsolationForest/LOF/AutoEncoder)',
                       '클래스 불균형 대응(SMOTE/class_weight)', 'XAI(SHAP)'],
        'leakage': ['failure_type'],   # is_anomaly 와 직접 연동된 라벨
        'eda_notes': [
            '보안위협 시 connections_per_ip 급증(정상 대비 수십~수백 배) → 강력한 분류 피처',
            '장비장애 시 cpu_usage 급락 + packet_loss_rate 급등 → 고유 패턴',
            '과부하는 cpu/memory/bandwidth 모두 85%+ → 복합 지표 기반 탐지',
            '이상 비율 낮음(약 3~5%) → 클래스 불균형 심각, Accuracy 단독 평가 금지',
        ],
        'columns': {
            'timestamp': ('수집 일시', 'DATETIME', '5분 단위'),
            'node_id': ('노드 ID', 'VARCHAR', 'NODE-001~NODE-020'),
            'node_type': ('노드 유형', 'VARCHAR', '라우터/스위치/서버/방화벽'),
            'zone': ('네트워크 구역', 'VARCHAR', 'DMZ/내부망/지휘망/연동망'),
            'cpu_usage': ('CPU 사용률(%)', 'FLOAT', '0~100'),
            'memory_usage': ('메모리 사용률(%)', 'FLOAT', '0~100'),
            'bandwidth_utilization': ('대역폭 사용률(%)', 'FLOAT', '0~100'),
            'packet_count': ('패킷 수', 'INT', '5분간 집계'),
            'packet_loss_rate': ('패킷 손실률(%)', 'FLOAT', '0~100'),
            'latency_ms': ('응답 지연시간(ms)', 'FLOAT', ''),
            'error_count': ('에러 발생 건수', 'INT', '5분간 집계'),
            'session_count': ('활성 세션 수', 'INT', ''),
            'traffic_volume_mb': ('트래픽량(MB)', 'FLOAT', '5분간 집계'),
            'connections_per_ip': ('IP당 연결 수', 'FLOAT', '보안위협 분류 핵심 피처'),
            'is_anomaly': ('이상 여부', 'INT', '0/1, ★타겟★. 이상비율 약 3~5%'),
            'failure_type': ('장애 유형', 'VARCHAR', '정상/과부하/장비장애/보안위협'),
        },
    },
]


def detect_profile(columns):
    """컬럼 시그니처로 알려진 데이터셋 프로파일을 찾는다. 없으면 None(범용 모드)."""
    s = set(columns)
    for p in PROFILES:
        if set(p['signature']) <= s:
            return p
    return None


def suggest_target(df, profile=None):
    """타깃 컬럼 자동 추천.
    1) 알려진 프로파일의 타깃  2) 흔한 타깃 이름  3) 이진 컬럼  4) 마지막 컬럼.
    """
    cols = list(df.columns)
    if profile and profile.get('target') in cols:
        return profile['target']
    low = {c.lower(): c for c in cols}
    for name in COMMON_TARGET_NAMES:
        if name in low:
            return low[name]
    for c in cols:
        try:
            if df[c].nunique(dropna=True) == 2:
                return c
        except TypeError:
            continue
    return cols[-1]


def column_help(profile, col):
    """컬럼 설명 도움말 문자열 (프로파일에 있으면) — selectbox help / 표 설명용."""
    if profile and col in profile.get('columns', {}):
        desc, typ, note = profile['columns'][col]
        return f'{desc} ({typ})' + (f' · {note}' if note else '')
    return None


def columns_table(profile, df_columns):
    """현재 df 컬럼에 대한 설명/타입/비고 매핑 (없는 컬럼은 빈 값)."""
    rows = {}
    cols = profile.get('columns', {}) if profile else {}
    for c in df_columns:
        if c in cols:
            desc, typ, note = cols[c]
        else:
            desc, typ, note = ('', '', '')
        rows[c] = {'설명': desc, '정의타입': typ, '비고': note}
    return rows
