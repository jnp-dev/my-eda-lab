"""
장비 정비이력 EDA → 모델링 → 예측 통합 Lab (Streamlit)
=======================================================
'장비정비_실습가이드.ipynb' 의 실습 1~4 전 과정을 화면에서 실행하면서
코드와 결과를 함께 점검할 수 있는 앱.

특징
- 데이터가 바뀌어도 자동 적응: 파일 업로드 + 시트 선택 + 타깃/컬럼 자동 감지
- 각 단계마다 '코드 보기' 로 노트북 대응 코드를 함께 표시
- 전처리·다중모델 비교·Optuna 튜닝·SHAP 해석·실시간 예측까지 일괄 제공

실행:  streamlit run eda_lab_app.py
"""
import os
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import dataset_profiles as dp   # 알려진 데이터셋 메타데이터(설명/추천 타깃) — 없으면 범용 모드

# 무해한 경고 억제 (결과엔 영향 없음 — 데모 화면 정리용)
#  · sklearn: DataFrame 으로 학습 후 ndarray 로 호출 시의 피처명 경고 (SHAP 등)
#  · 수렴/버전 관련 잡음
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 한글 폰트 (macOS). 환경에 맞게 자동 폴백.
KFONT = None
KFONT_PATH = None
for _f in ['AppleGothic', 'Apple SD Gothic Neo', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']:
    _hits = [f.fname for f in matplotlib.font_manager.fontManager.ttflist if _f in f.name]
    if _hits:
        KFONT, KFONT_PATH = _f, _hits[0]
        break


def apply_korean_font():
    """matplotlib 한글 폰트 적용. font.sans-serif 맨 앞에도 등록해
    ydata-profiling/seaborn 이 family 를 'sans-serif' 로 되돌려도 한글이 유지되게 한다."""
    if KFONT:
        plt.rcParams['font.family'] = KFONT
        existing = [x for x in plt.rcParams.get('font.sans-serif', []) if x != KFONT]
        plt.rcParams['font.sans-serif'] = [KFONT] + existing
    plt.rcParams['axes.unicode_minus'] = False


apply_korean_font()

st.set_page_config(page_title='데이터 분석 Lab — EDA · 모델링 · 예측',
                   page_icon=':material/insights:', layout='wide')

# ── Tailwind 디자인 토큰 기반 프로페셔널 테마 (Inter 폰트 + slate/indigo 팔레트) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root{
  --bg:#f8fafc; --surface:#ffffff; --border:#e2e8f0;
  --text:#0f172a; --muted:#64748b; --primary:#4f46e5; --primary-d:#4338ca;
}
html, body, [class*="css"]{ font-family:'Inter','Pretendard','Apple SD Gothic Neo',sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }
/* 기본 본문/위젯 글자색 강제 (다크모드에서도 가독성 확보) */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"]{ color:var(--text); }
.stApp .stCaption, .stApp small, [data-testid="stCaptionContainer"]{ color:var(--muted); }
/* 본문: 최소 넓이 보장 + 최대한 넓게 */
.block-container{ padding-top:1.4rem; padding-left:2.4rem; padding-right:2.4rem;
  max-width:none; min-width:820px; }
h1,h2,h3,h4{ color:var(--text); font-weight:700; letter-spacing:-0.01em; }
/* 모든 제목(타이틀/헤더/서브헤더)을 동일 크기로 통일 */
h1, h2, h3, h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
[data-testid="stHeading"] h3, [data-testid="stHeading"] h4{
  font-size:1.25rem !important; font-weight:700 !important;
}

/* 메트릭 → 카드 */
[data-testid="stMetric"]{
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:1rem 1.1rem;
  box-shadow:0 1px 2px rgba(15,23,42,.05);
}
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:500; }
[data-testid="stMetricValue"]{ color:var(--text); font-weight:700; }

/* 버튼 */
.stButton>button, .stDownloadButton>button{
  background:var(--primary); color:#fff !important; border:0; border-radius:10px;
  padding:.5rem 1.15rem; font-weight:600; transition:.15s;
}
/* 버튼 내부 글자(텍스트/마크다운) 모두 흰색 강제 */
.stButton>button *, .stDownloadButton>button *{ color:#fff !important; }
.stButton>button:hover{ background:var(--primary-d); transform:translateY(-1px);
  box-shadow:0 4px 12px rgba(79,70,229,.25); }

/* 사이드바 */
[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
/* 사이드바 안 모든 글자색 강제 — 흰 배경 위 가독성 */
[data-testid="stSidebar"] *{ color:var(--text); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{ color:var(--text); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small{ color:var(--muted); }
[data-testid="stSidebar"] [role="radiogroup"]>label{
  padding:.4rem .55rem; border-radius:8px; font-weight:500; transition:.12s;
}
[data-testid="stSidebar"] [role="radiogroup"]>label:hover{ background:#f1f5f9; }
/* 입력/셀렉트 박스 — 흰 배경 + 짙은 글자 */
[data-testid="stSidebar"] [data-baseweb="select"] input{ color:var(--text); }
[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:#fff; border:1px solid var(--border);
}
/* 멀티선택 태그(칩) — 옅은 보라 배경 + 흰색 bold 글자 */
[data-baseweb="tag"]{
  background:var(--primary) !important; border-radius:8px !important;
}
[data-baseweb="tag"] span, [data-baseweb="tag"] *{
  color:#fff !important; font-weight:700 !important;
}
/* 태그 삭제(x) 아이콘도 흰색으로 또렷하게 */
[data-baseweb="tag"] svg{ fill:#fff !important; color:#fff !important; }

/* 상단 페이지 탭 바 (메인 영역 가로 라디오) */
/* 래퍼들이 높이를 제한해 글자를 자르지 않도록 overflow 해제 */
.st-key-pagenav, .st-key-pagenav *{ overflow:visible !important; }
.st-key-pagenav [data-testid="stRadio"],
.st-key-pagenav [data-testid="stRadio"]>div{ height:auto !important; }

.st-key-pagenav [role="radiogroup"]{
  display:flex; flex-wrap:wrap; gap:.25rem; border-bottom:2px solid var(--border);
  margin-top:1.4rem; margin-bottom:1.1rem; padding:0; align-items:flex-end;
}
.st-key-pagenav [role="radiogroup"]>label{
  margin:0 !important; padding:1.1rem 1.2rem .85rem !important;
  border-radius:10px 10px 0 0;
  font-weight:700; color:var(--text); cursor:pointer;
  border:1px solid transparent; border-bottom:none;
  transition:.12s;
  display:flex !important; align-items:center !important;
  justify-content:center; min-height:3.2rem !important; height:auto !important;
}
/* 라벨 안 텍스트 컨테이너 — 자체 높이/정렬 제거, 세로 중앙 */
.st-key-pagenav [role="radiogroup"]>label>div{
  display:flex !important; align-items:center !important;
  height:auto !important; min-height:0 !important; line-height:1.8 !important;
}
.st-key-pagenav [role="radiogroup"]>label *{
  color:var(--text); line-height:1.8 !important; height:auto !important;
  vertical-align:middle !important;
  font-size:1.25rem !important; font-weight:700 !important;
}
.st-key-pagenav [role="radiogroup"]>label p{
  margin:0 !important; line-height:1.8 !important;
  font-size:1.25rem !important; font-weight:700 !important;
}
.st-key-pagenav [role="radiogroup"]>label:hover{ background:#eef2ff; color:var(--primary); }
.st-key-pagenav [role="radiogroup"]>label:hover *{ color:var(--primary); }
/* 선택된 탭 — 보라 배경 + 흰 글씨 */
.st-key-pagenav [role="radiogroup"]>label:has(input:checked){
  background:var(--primary); border-color:var(--primary);
}
.st-key-pagenav [role="radiogroup"]>label:has(input:checked),
.st-key-pagenav [role="radiogroup"]>label:has(input:checked) *{
  color:#fff !important;
}
/* 라디오 동그라미 점 숨김 (탭처럼 보이게) */
.st-key-pagenav [role="radiogroup"]>label>div:first-child{ display:none; }

/* 익스팬더 / 데이터프레임 / 구분선 */
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:12px; background:var(--surface); }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; }
hr{ border-color:var(--border); }
</style>
""", unsafe_allow_html=True)

DEFAULT_PATH = 'data/dataset1_equipment_maintenance.xlsx'
ARTIFACT_DIR = 'artifacts'
PRED_LOG = 'predictions.csv'


# ──────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────
def show_code(code: str, label: str = ':material/code: 코드 보기'):
    """단계별 대응 코드를 펼침 박스로 표시 — '코드 점검' 목적."""
    with st.expander(label):
        st.code(code, language='python')


def insight(body: str, title: str = '인사이트', kind: str = 'info', expanded: bool = True):
    """데이터에서 계산한 해석/인사이트를 펼침 카드로 표시 — '결과를 어떻게 읽는가'를 설명.
    실습가이드의 '📈 인사이트' 블록을 UI 로 옮긴 것. body 는 markdown(여러 줄/불릿 가능)."""
    icon = {'info': '💡', 'warn': '⚠️', 'good': '✅', 'idea': '🔎'}.get(kind, '💡')
    with st.expander(f'{icon} {title}', expanded=expanded):
        st.markdown(body)


def id_like_columns(df: pd.DataFrame) -> list[str]:
    """개체(그룹) 식별자로 보이는 컬럼 — 같은 개체가 여러 행에 반복 등장하는 컬럼.
    그룹 누수 방지용 분할(GroupShuffleSplit/GroupKFold)의 그룹 기준 후보로 쓴다.
    이름이 id/번호/코드 인 컬럼을 우선하고, 개체당 평균 행 수가 충분한 것만 남긴다."""
    n = len(df)
    scored = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            continue
        nun = df[c].nunique(dropna=True)
        if nun <= 1 or nun >= n:          # 상수 / 행마다 고유(식별자 아님) 제외
            continue
        rows_per = n / nun
        if rows_per < 2:                   # 개체당 평균 2행 미만이면 그룹 의미가 약함
            continue
        name = str(c).lower()
        score = 0.0
        if 'id' in name or '아이디' in name:
            score += 100
        if '번호' in name:
            score += 80
        if 'code' in name or '코드' in name:
            score += 40
        if df[c].dtype == object:
            score += 10
        score += min(nun, 1000) / 50.0     # 개체 수가 많을수록 '식별자다움' 가산
        if score >= 10 or rows_per >= 3:
            scored.append((score, nun, c))
    scored.sort(key=lambda x: (-x[0], -x[1]))   # 점수↓, 개체수↓ 순
    return [c for _, _, c in scored]


def imbalance_info(y: pd.Series):
    """이진 분류 타깃의 불균형 정도 → (소수클래스비율%, scale_pos_weight) 반환.
    이진이 아니면 (None, None)."""
    vc = y.value_counts()
    if len(vc) != 2:
        return None, None
    minor = vc.min()
    major = vc.max()
    minor_pct = minor / vc.sum() * 100
    spw = major / minor if minor else 1.0
    return minor_pct, spw


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes | None, path: str):
    """엑셀 시트 목록 (csv면 None)."""
    src = file_bytes if file_bytes is not None else path
    try:
        return pd.ExcelFile(src).sheet_names
    except Exception:
        return None


@st.cache_data(show_spinner=True)
def load_data(file_bytes: bytes | None, path: str, sheet, is_csv: bool):
    """데이터 로드 + 날짜형 자동 파싱. 업로드/기본경로 모두 지원."""
    src = file_bytes if file_bytes is not None else path
    if is_csv:
        df = pd.read_csv(src if file_bytes is None else __import__('io').BytesIO(file_bytes))
    else:
        df = pd.read_excel(src, sheet_name=sheet)
    # 날짜형 자동 감지/변환 (date/time/datetime/timestamp/일시/시각)
    date_hints = ('date', 'time', 'datetime', 'timestamp', '일시', '시각')
    for c in df.columns:
        if df[c].dtype == object and any(h in str(c).lower() for h in date_hints):
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().mean() > 0.9:   # 대부분 파싱되면 날짜로 채택
                    df[c] = parsed
            except Exception:
                pass
    return df


def infer_task(y: pd.Series) -> str:
    """타깃으로부터 분류/회귀 자동 판별."""
    if y.dtype == object or str(y.dtype).startswith('category'):
        return 'classification'
    if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
        return 'regression'
    return 'classification' if y.nunique() <= 20 else 'regression'


def numeric_target(s: pd.Series):
    """타깃을 rate 계산용 수치 Series 로 변환.
    반환: (수치 Series 또는 None, 양성 라벨 또는 None).
    - 수치형          → 그대로 (양성 라벨 None)
    - 범주형 + 이진    → 소수 클래스를 1 로 인코딩 (양성 라벨 반환)
    - 범주형 + 다중    → (None, None)  ← 단일 평균 불가
    """
    if pd.api.types.is_numeric_dtype(s):
        return s, None
    if s.nunique(dropna=True) == 2:
        pos = s.value_counts().index[-1]   # 빈도 낮은 쪽을 양성으로 간주
        return (s == pos).astype(int), pos
    return None, None


def auto_drop_candidates(df: pd.DataFrame, target: str) -> list[str]:
    """식별자/날짜 등 모델 피처로 부적절한 컬럼 자동 추천."""
    drop = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        if np.issubdtype(df[c].dtype, np.datetime64):
            drop.append(c)
        elif df[c].dtype == object and df[c].nunique() > max(50, 0.5 * n):
            drop.append(c)  # 고유값이 너무 많은 object → 식별자로 간주
    return drop


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """수치형=중앙값, 범주형='N/A' 로 결측 대체 (데이터 무관 generic)."""
    out = df.copy()
    for c in out.columns:
        if out[c].isnull().any():
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            else:
                out[c] = out[c].fillna('N/A')
    return out


# ── 전처리 계획 (진단 → 결정 → 적용) ───────────────────────────────
NULL_OPTS = ['대체안함', '중앙값', '평균', '최빈값', '상수(N/A·0)', '앞값(ffill)', '행 제거']
SCALE_OPTS = ['없음', '표준(Standard)', 'Min-Max', 'Robust', 'Log']
BIN_OPTS = ['없음', '동일폭', '분위수']


def _scaler_for(name):
    """이름 → sklearn 스케일러 (없음=passthrough)."""
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                       RobustScaler, FunctionTransformer)
    from sklearn.pipeline import Pipeline
    if name == 'Min-Max':
        return MinMaxScaler()
    if name == 'Robust':
        return RobustScaler()
    if name == 'Log':
        return Pipeline([('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                         ('sc', StandardScaler())])
    if name == '없음':
        return 'passthrough'
    return StandardScaler()


def _num_imputer(name):
    """결측 대체 전략 → 수치형 SimpleImputer (행제거/ffill/대체안함은 사전처리되므로 median 폴백)."""
    from sklearn.impute import SimpleImputer
    if name == '평균':
        return SimpleImputer(strategy='mean')
    if name == '최빈값':
        return SimpleImputer(strategy='most_frequent')
    if name == '상수(N/A·0)':
        return SimpleImputer(strategy='constant', fill_value=0)
    return SimpleImputer(strategy='median')   # 중앙값/대체안함/ffill/행제거 폴백


def _cat_imputer(name):
    """결측 대체 전략 → 범주형 SimpleImputer."""
    from sklearn.impute import SimpleImputer
    if name == '상수(N/A·0)':
        return SimpleImputer(strategy='constant', fill_value='N/A')
    return SimpleImputer(strategy='most_frequent')   # 최빈값/그외 폴백


def make_preprocessor(num_cols, cat_cols, plan=None):
    """전처리 계획 전체(결측대체·범주화·스케일링·인코딩)를 sklearn 파이프라인으로 구성.
    컬럼별 Pipeline 을 ColumnTransformer 로 묶어, fit 은 학습 데이터에서만 일어나
    데이터 누수가 없고 model.pkl 하나로 예측까지 일관되게 동작한다.
    plan 이 없으면: 수치=median+StandardScaler, 범주=최빈값+OneHot (안전한 기본 전처리)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
    from sklearn.compose import ColumnTransformer
    lut = {r['컬럼']: r for r in plan} if plan else {}
    transformers = []

    for c in num_cols:
        rec = lut.get(c, {})
        imp = _num_imputer(rec.get('결측대체'))
        binm = rec.get('범주화', '없음')
        if binm and binm != '없음':
            nb = int(rec.get('bins', 5) or 5)
            strat = 'uniform' if binm == '동일폭' else 'quantile'
            steps = [('impute', imp),
                     ('bin', KBinsDiscretizer(n_bins=nb, encode='onehot-dense',
                                              strategy=strat, subsample=None))]
        else:
            steps = [('impute', imp), ('scale', _scaler_for(rec.get('스케일링', '표준(Standard)')))]
        transformers.append((f'n_{c}', Pipeline(steps), [c]))

    for c in cat_cols:
        rec = lut.get(c, {})
        transformers.append((f'c_{c}', Pipeline([
            ('impute', _cat_imputer(rec.get('결측대체'))),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), [c]))

    return ColumnTransformer(transformers)


def prepare_rows(base: pd.DataFrame, plan: list[dict], target: str) -> pd.DataFrame:
    """행 단위 사전 처리만 수행: 컬럼 제외(사용 안 함) + '행 제거'/'앞값(ffill)' 결측 전략.
    (결측대체·범주화·스케일링은 파이프라인에서 처리하므로 여기서 하지 않는다.)"""
    out = base.copy()
    lut = {r['컬럼']: r for r in plan}
    for c, rec in lut.items():
        if c in out.columns and c != target and not rec.get('사용', True):
            out = out.drop(columns=[c])
    for c, rec in lut.items():
        if c not in out.columns:
            continue
        strat = rec.get('결측대체')
        if strat == '행 제거' and out[c].isnull().any():
            out = out.dropna(subset=[c])
        elif strat == '앞값(ffill)' and out[c].isnull().any():
            out[c] = out[c].ffill().bfill()
    return out


def recommend_column(s, is_target):
    """컬럼 통계를 보고 결측/스케일/범주화 추천을 반환."""
    null_pct = float(s.isnull().mean() * 100)
    nun = int(s.nunique(dropna=True))
    is_num = pd.api.types.is_numeric_dtype(s)
    is_dt = np.issubdtype(s.dtype, np.datetime64)
    skew = 0.0
    if is_num and not is_dt and nun > 2:
        try:
            skew = float(abs(s.dropna().skew()))
        except Exception:
            skew = 0.0

    # 결측 대체
    if null_pct == 0:
        null_rec = '대체안함'
    elif is_num and not is_dt:
        null_rec = '중앙값' if skew > 1 else '평균'
    else:
        null_rec = '최빈값'

    # 스케일링 (수치형만)
    if is_num and not is_dt:
        nonneg = bool(s.dropna().min() >= 0) if s.notna().any() else False
        scale_rec = 'Log' if (skew > 1.5 and nonneg) else '표준(Standard)'
    else:
        scale_rec = '없음'

    # 사용 여부: 타깃·날짜는 기본 제외
    use_rec = not (is_target or is_dt)
    kind = '날짜' if is_dt else ('수치' if is_num else '범주')
    return {
        'null_pct': round(null_pct, 1), 'nunique': nun, 'kind': kind,
        'use': use_rec, 'null': null_rec, 'scale': scale_rec,
        'bin': '없음', 'bins': 5, 'skew': round(skew, 2),
    }


def build_default_plan(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """컬럼별 진단 + 추천을 담은 편집용 계획 DataFrame."""
    rows = []
    for c in df.columns:
        r = recommend_column(df[c], is_target=(c == target))
        rows.append({
            '컬럼': c, '타입': r['kind'], '타깃': (c == target),
            'null%': r['null_pct'], '고유값': r['nunique'], '왜도': r['skew'],
            '추천': f"{r['null']} / {r['scale']}"
                    + ('' if r['bin'] == '없음' else f" / {r['bin']}"),
            '사용': r['use'], '결측대체': r['null'],
            '스케일링': r['scale'], '범주화': r['bin'], 'bins': r['bins'],
        })
    return pd.DataFrame(rows)


def apply_plan(base: pd.DataFrame, plan: list[dict], target: str):
    """계획(plan)에 따라 결측 대체·범주화·컬럼제외를 base 에 적용.
    반환: (변환된 df, {컬럼:스케일러이름} 스케일 맵)."""
    out = base.copy()
    scale_map = {}
    for rec in plan:
        c = rec.get('컬럼')
        if c is None or c not in out.columns:
            continue
        is_tgt = (c == target)
        # 사용 안 함 → 제외 (타깃은 항상 유지)
        if not is_tgt and not rec.get('사용', True):
            out = out.drop(columns=[c])
            continue
        s = out[c]
        is_num = pd.api.types.is_numeric_dtype(s)

        # 결측 대체
        strat = rec.get('결측대체', '대체안함')
        if s.isnull().any() and strat != '대체안함':
            if strat == '중앙값' and is_num:
                out[c] = s.fillna(s.median())
            elif strat == '평균' and is_num:
                out[c] = s.fillna(s.mean())
            elif strat == '최빈값':
                m = s.mode(dropna=True)
                out[c] = s.fillna(m.iloc[0] if len(m) else ('N/A' if not is_num else 0))
            elif strat == '상수(N/A·0)':
                out[c] = s.fillna(0 if is_num else 'N/A')
            elif strat == '앞값(ffill)':
                out[c] = s.ffill().bfill()
            elif strat == '행 제거':
                out = out.dropna(subset=[c])
            s = out[c]
            is_num = pd.api.types.is_numeric_dtype(s)

        # 범주화 (수치형, 타깃 제외) → 구간 라벨(범주형)로 변환
        binm = rec.get('범주화', '없음')
        if not is_tgt and is_num and binm != '없음':
            nb = int(rec.get('bins', 5) or 5)
            try:
                if binm == '동일폭':
                    out[c] = pd.cut(out[c], bins=nb).astype(str)
                else:  # 분위수
                    out[c] = pd.qcut(out[c], q=nb, duplicates='drop').astype(str)
            except Exception:
                pass  # 범주화 실패 시 원본 유지
        elif not is_tgt and is_num:
            # 범주화 안 한 수치형만 스케일 대상
            sc = rec.get('스케일링', '없음')
            if sc and sc != '없음':
                scale_map[c] = sc
    return out, scale_map


def get_models(task: str, selected: list[str], balanced: bool = False, spw: float = 1.0):
    """비교용 모델 풀. balanced=True 면 불균형 보정(class_weight='balanced' / scale_pos_weight=spw)을 적용.
    soft 한 보정이라 다수 클래스만 찍는 것을 막고 소수 클래스 Recall 을 끌어올린다."""
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                  RandomForestRegressor, GradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.dummy import DummyClassifier
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    cw = 'balanced' if balanced else None
    pw = spw if balanced else 1.0
    if task == 'classification':
        pool = {
            'Dummy(baseline)': DummyClassifier(strategy='most_frequent'),
            'Logistic':     LogisticRegression(max_iter=1000, class_weight=cw),
            'GaussianNB':   GaussianNB(),   # class_weight 미지원(비교용)
            'RandomForest': RandomForestClassifier(random_state=42, class_weight=cw),
            'GBM':          GradientBoostingClassifier(random_state=42),  # class_weight 미지원
            'XGBoost':      XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=pw),
            'LightGBM':     LGBMClassifier(random_state=42, verbose=-1, class_weight=cw),
        }
    else:
        pool = {
            'Linear':       LinearRegression(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'GBM':          GradientBoostingRegressor(random_state=42),
            'XGBoost':      XGBRegressor(random_state=42),
            'LightGBM':     LGBMRegressor(random_state=42, verbose=-1),
        }
    return {k: v for k, v in pool.items() if k in selected}


# ──────────────────────────────────────────────────────────────────
# 사이드바 — 데이터 소스 & 설정 (모든 페이지 공유)
# ──────────────────────────────────────────────────────────────────
st.sidebar.title(':material/settings: 데이터 & 설정')

up = st.sidebar.file_uploader('데이터 업로드 (.xlsx / .csv)', type=['xlsx', 'csv'])
file_bytes = up.getvalue() if up is not None else None
path = DEFAULT_PATH
# CSV 여부: 업로드면 업로드 파일명, 아니면 기본 경로 확장자로 판단 (범용 데이터 지원)
is_csv = (up.name.lower().endswith('.csv') if up is not None
          else str(path).lower().endswith('.csv'))

if file_bytes is None and not os.path.exists(DEFAULT_PATH):
    st.sidebar.warning('기본 데이터가 없습니다. 파일을 업로드하세요.')
    st.stop()

sheet = None
if not is_csv:
    sheets = list_sheets(file_bytes, path)
    if sheets:
        # '데이터' 시트가 있으면 기본 선택 (컬럼정의서 회피)
        default_idx = sheets.index('데이터') if '데이터' in sheets else len(sheets) - 1
        sheet = st.sidebar.selectbox('시트 선택', sheets, index=default_idx)

df = load_data(file_bytes, path, sheet, is_csv)

# 알려진 데이터셋이면 프로파일 인식 (없으면 None → 범용 모드)
profile = dp.detect_profile(df.columns)
if profile:
    st.sidebar.caption(f":material/verified: 인식된 데이터셋 — **{profile['name']}**")
else:
    st.sidebar.caption(':material/dataset: 범용 분석 모드 (임의 데이터)')

# 타깃 자동 추천 (프로파일 → 흔한 타깃명 → 이진 컬럼 → 마지막 컬럼)
cols = df.columns.tolist()
default_target = dp.suggest_target(df, profile)
target = st.sidebar.selectbox(
    ':material/flag: 타깃 변수', cols, index=cols.index(default_target),
    help=dp.column_help(profile, default_target) or '예측할 대상(정답) 컬럼을 선택하세요.')
task = infer_task(df[target])
st.sidebar.caption(f'자동 판별 문제 유형: **{task}**  (타깃 고유값 {df[target].nunique()}개)')

drop_default = auto_drop_candidates(df, target)
drop_cols = st.sidebar.multiselect('모델에서 제외할 컬럼 (식별자/날짜 등)',
                                   [c for c in cols if c != target], default=drop_default)

# 누수 의심 컬럼 토글 (프로파일에 정의된 경우만)
leak_known = [c for c in (profile['leakage'] if profile else []) if c in cols]
exclude_leak = False
if leak_known:
    exclude_leak = st.sidebar.checkbox(f'누수 의심 컬럼 제외 ({", ".join(leak_known)})', value=False)

# 그룹(개체) 컬럼 — 같은 개체(예: 장비)가 여러 행에 반복 등장하면, 무작위 분할 시
# 같은 개체가 train/test 양쪽에 들어가는 '그룹 누수'가 생긴다. 이를 막을 분할 기준.
group_candidates = id_like_columns(df)
group_col = st.sidebar.selectbox(
    ':material/groups: 그룹(개체) 컬럼 — 누수 방지 분할 기준',
    ['(사용 안 함)'] + cols,
    index=(cols.index(group_candidates[0]) + 1) if group_candidates else 0,
    help='같은 개체가 여러 정비/관측 행으로 반복 등장할 때 선택하세요. '
         '선택하면 모델링에서 GroupShuffleSplit·GroupKFold 로 "같은 개체는 한쪽에만" 분할해 '
         '그룹 누수를 막습니다. (장비 정비 데이터: equipment_id)')
group_col = None if group_col == '(사용 안 함)' else group_col

# EDA·시각화에 전처리 계획 적용 여부 (결측대체+범주화 반영된 데이터로 탐색)
st.sidebar.divider()
eda_apply_prep = st.sidebar.checkbox(
    'EDA·시각화에 전처리 계획 적용',
    value=False, key='eda_apply_prep',
    help='켜면 결측 대체·범주화가 반영된 데이터로 EDA/시각화를 봅니다. '
         '(스케일링은 가독성을 위해 적용하지 않음) — 모델링은 별도로 항상 계획을 반영합니다.')


def get_view_df():
    """EDA·시각화용 데이터. 토글이 켜져 있고 계획이 있으면 결측대체·범주화 적용본을 반환."""
    plan = st.session_state.get('prep_plan')
    if eda_apply_prep and plan:
        try:
            view, _ = apply_plan(df, plan, target)
            return view
        except Exception:
            return df
    return df


# 세션 공유용
st.session_state['df'] = df
st.session_state['target'] = target
st.session_state['task'] = task

_PAGE_ICONS = {
    '0. 데이터 개요': 'dataset',
    '1. EDA': 'query_stats',
    '2. 전처리 계획': 'cleaning_services',
    '3. 종합 대시보드': 'dashboard',
    '4. 모델링 · 평가 · 해석': 'model_training',
    '5. 예측 앱': 'online_prediction',
    '6. 시각화 레시피·코드': 'auto_graph',
}
# 페이지 선택 — 메인 영역 상단 탭 바
with st.container(key='pagenav'):
    page = st.radio(
        '페이지', list(_PAGE_ICONS),
        format_func=lambda k: f':material/{_PAGE_ICONS[k]}: {k}',
        horizontal=True, label_visibility='collapsed')


def get_feature_frame():
    """사이드바 설정 + 전처리 계획을 반영한 X, y, num/cat 반환 (전 페이지 공통).
    결측대체·범주화·스케일링은 파이프라인(make_preprocessor)에서 처리하므로,
    여기서는 행 단위 사전 처리(컬럼 제외·행 제거·ffill)만 적용한다. X 에는 결측이 남아 있을 수 있다."""
    drop = list(drop_cols) + (leak_known if exclude_leak else [])
    base = df.drop(columns=[c for c in drop if c in df.columns])

    plan = st.session_state.get('prep_plan')
    work = prepare_rows(base, plan, target) if plan else base

    X = work.drop(columns=[target])
    # datetime 잔여 컬럼 제거 (인코딩 불가)
    dt_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    X = X.drop(columns=dt_cols)
    y = work[target]
    # 타깃 결측 행 제거 (학습 불가)
    if y.isnull().any():
        keep = y.notna()
        X, y = X[keep], y[keep]
    num_cols = X.select_dtypes('number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()
    return X, y, num_cols, cat_cols


# ══════════════════════════════════════════════════════════════════
# 0. 데이터 개요
# ══════════════════════════════════════════════════════════════════
if page == '0. 데이터 개요':
    st.title(':material/dataset: 데이터 개요')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('행', f'{len(df):,}')
    c2.metric('열', f'{df.shape[1]}')
    c3.metric('타깃', target)
    c4.metric('문제 유형', task)

    # 알려진 데이터셋이면 학습목표/추천기법 안내 (가이드 반영). 모르면 범용 안내.
    if profile:
        st.info(f":material/menu_book: **{profile['name']}**  ·  {profile['category']}\n\n"
                f"**학습 목표** — {profile['objective']}")
        st.markdown('**추천 기법**: ' + ' · '.join(f'`{t}`' for t in profile['techniques']))
        if profile.get('target') in df.columns:
            st.caption(f"가이드 추천 타깃: `{profile['target']}` "
                       f"(현재 선택: `{target}`) — 사이드바에서 변경 가능")
    else:
        st.caption('일반(범용) 데이터로 인식했습니다. 사이드바에서 타깃만 지정하면 EDA·모델링·예측이 동작합니다.')

    st.subheader('컬럼 정보 (설명 · 타입 · 고유값 · 결측)')
    meta = dp.columns_table(profile, df.columns)   # 알려진 컬럼이면 설명/비고 포함
    info = pd.DataFrame({
        '설명': {c: meta[c]['설명'] for c in df.columns},
        'dtype': df.dtypes.astype(str),
        'nunique': df.nunique(),
        'null_%': (df.isnull().mean() * 100).round(1),
        '비고': {c: meta[c]['비고'] for c in df.columns},
    })
    # 설명/비고가 전부 비어있으면(범용 데이터) 해당 컬럼 숨김
    if info['설명'].str.len().sum() == 0:
        info = info.drop(columns=['설명', '비고'])
    st.dataframe(info, width='stretch')

    st.subheader('데이터 미리보기')
    pc1, pc2 = st.columns([1, 2])
    with pc1:
        view_mode = st.radio('보기 방식', ['미리보기(상위 N행)', '전체 보기'],
                             horizontal=True, key='ov_view_mode')
    with pc2:
        sel_cols = st.multiselect('표시할 컬럼 (비우면 전체)', list(df.columns),
                                  default=[], key='ov_cols')

    fc1, fc2 = st.columns([1, 2])
    with fc1:
        n_rows = st.number_input('미리볼 행 수', min_value=5, max_value=len(df),
                                 value=min(20, len(df)), step=5, key='ov_nrows',
                                 disabled=(view_mode == '전체 보기'))
    with fc2:
        query = st.text_input('검색 (모든 컬럼 텍스트 포함 일치)', '', key='ov_query',
                              placeholder='예: EQ-0001, 발전기, UNIT-01 ...')

    view = df if not sel_cols else df[sel_cols]
    if query:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        view = view[mask]

    shown = view if view_mode == '전체 보기' else view.head(int(n_rows))
    st.caption(f'표시 {len(shown):,}행 / 필터 후 {len(view):,}행 / 전체 {len(df):,}행')
    st.dataframe(shown, width='stretch')
    st.download_button('필터 결과 CSV 다운로드',
                       view.to_csv(index=False).encode('utf-8-sig'),
                       file_name='preview_filtered.csv', mime='text/csv',
                       key='ov_download')

    show_code(
        "df = pd.read_excel(path, sheet_name='데이터')\n"
        "df.shape, df.dtypes, df.isnull().sum()")


# ══════════════════════════════════════════════════════════════════
# 전처리 계획 (진단 → 결정 → 적용)
# ══════════════════════════════════════════════════════════════════
elif page == '2. 전처리 계획':
    st.title(':material/cleaning_services: 실습 2 — 전처리 계획 (진단 · 결정 · 적용)')
    st.caption('모델링 전에 컬럼별 **결측 대체 · 스케일링 · 범주화**를 결정합니다. '
               '아래 표를 수정하고 **[계획 적용]** 을 누르면 EDA·시각화·모델링에 반영됩니다.')

    applied = st.session_state.get('prep_plan')
    if applied:
        st.success(f':material/check_circle: 현재 적용된 계획이 있습니다 ({len(applied)}개 컬럼). '
                   '표를 다시 수정 후 [계획 적용]을 누르면 갱신됩니다.')
    else:
        st.info(':material/info: 아직 적용된 계획이 없습니다. 지금은 자동 전처리'
                '(수치=중앙값, 범주=N/A, 표준 스케일링)가 사용됩니다.')

    # ── STEP 1 · 진단 (자동 추천) ──
    st.subheader('STEP 1 · 진단 & 추천')
    plan_df = build_default_plan(df, target)
    diag = plan_df[['컬럼', '타입', '타깃', 'null%', '고유값', '왜도', '추천']]
    st.dataframe(diag, width='stretch', hide_index=True)
    st.caption('왜도(skew) 절댓값이 크면(>1.5, 음수 없음) Log 변환을, 결측이 있으면 '
               '수치=중앙값/평균·범주=최빈값을 추천합니다. 추천은 기본값일 뿐 자유롭게 바꾸세요.')

    # ── STEP 2 · 결정 (편집) ──
    st.subheader('STEP 2 · 결정 (컬럼별 전처리 선택)')
    editable = plan_df[['컬럼', '타입', 'null%', '사용',
                        '결측대체', '스케일링', '범주화', 'bins']].copy()
    edited = st.data_editor(
        editable, width='stretch', hide_index=True, key='prep_editor',
        column_config={
            '컬럼': st.column_config.TextColumn('컬럼', disabled=True),
            '타입': st.column_config.TextColumn('타입', disabled=True, width='small'),
            'null%': st.column_config.NumberColumn('null%', disabled=True,
                                                   format='%.1f', width='small'),
            '사용': st.column_config.CheckboxColumn('모델 사용'),
            '결측대체': st.column_config.SelectboxColumn('결측 대체', options=NULL_OPTS),
            '스케일링': st.column_config.SelectboxColumn('스케일링', options=SCALE_OPTS),
            '범주화': st.column_config.SelectboxColumn('범주화', options=BIN_OPTS),
            'bins': st.column_config.NumberColumn('구간 수', min_value=2,
                                                  max_value=20, step=1, width='small'),
        })
    st.caption('· 범주화를 선택하면 해당 수치형은 구간(범주)으로 바뀌어 스케일링은 무시됩니다. '
               '· 스케일링은 누수 방지를 위해 모델 파이프라인 안에서 학습 데이터 기준으로 적용됩니다.')

    # ── STEP 3 · 적용 ──
    st.subheader('STEP 3 · 적용')
    b1, b2, b3 = st.columns([1, 1, 1])
    if b1.button(':material/play_arrow: 계획 적용', type='primary', key='prep_apply'):
        st.session_state['prep_plan'] = edited.to_dict('records')
        st.success('전처리 계획을 적용했습니다. EDA·시각화·모델링에 반영됩니다.')
        st.rerun()
    if b2.button(':material/restart_alt: 계획 초기화(자동 전처리)', key='prep_reset'):
        st.session_state.pop('prep_plan', None)
        st.session_state.pop('scale_map', None)
        st.success('계획을 초기화했습니다. 자동 전처리로 되돌립니다.')
        st.rerun()
    b3.download_button(':material/download: 계획 CSV',
                       edited.to_csv(index=False).encode('utf-8-sig'),
                       file_name='preprocess_plan.csv', mime='text/csv',
                       key='prep_csv')

    # 적용 결과 미리보기 (Before → After)
    with st.expander(':material/preview: 적용 결과 미리보기 (Before → After)', expanded=bool(applied)):
        drop = list(drop_cols) + (leak_known if exclude_leak else [])
        base = df.drop(columns=[c for c in drop if c in df.columns])
        after, scale_map = apply_plan(base, edited.to_dict('records'), target)
        cc1, cc2 = st.columns(2)
        cc1.markdown('**Before** (원본)')
        cc1.dataframe(base.head(10), width='stretch')
        cc2.markdown('**After** (계획 적용)')
        cc2.dataframe(after.head(10), width='stretch')
        rem_null = int(after.drop(columns=[target], errors='ignore').isnull().sum().sum())
        st.caption(f'적용 후 컬럼 {after.shape[1]}개 · 결측 잔여 {rem_null}개 · '
                   f'파이프라인 스케일링 대상 {len(scale_map)}개 '
                   + (f"({', '.join(f'{k}:{v}' for k, v in list(scale_map.items())[:6])}"
                      + (' …' if len(scale_map) > 6 else '') + ')' if scale_map else ''))

    show_code(
        "# 컬럼별 결측 대체\n"
        "df['col'] = df['col'].fillna(df['col'].median())   # 또는 mean / mode / 0\n"
        "# 범주화 (구간화)\n"
        "df['col_bin'] = pd.cut(df['col'], bins=5)           # 동일폭\n"
        "df['col_bin'] = pd.qcut(df['col'], q=5)             # 분위수\n"
        "# 스케일링 (학습 데이터 fit → 누수 방지)\n"
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n"
        "X_train_s = StandardScaler().fit_transform(X_train)")

    # ── 보조 1 · IQR 기반 이상치 탐지 ──────────────────────────────────
    st.divider()
    st.subheader(':material/troubleshoot: 이상치 탐지 (IQR · Tukey 규칙)')
    with st.expander('📦 IQR 이란? — 펼쳐서 개념 보기', expanded=False):
        st.markdown(
            '- **IQR = Q3 − Q1** : 데이터를 정렬했을 때 상위 75% 지점(**Q3**)과 하위 25% 지점(**Q1**) '
            '사이 거리 = **가운데 50% 가 퍼진 폭**. 평균·표준편차와 달리 **극단값에 둔감**해 이상치 기준으로 자주 쓰입니다.\n'
            '- **Tukey 규칙**: `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` 를 벗어나면 이상치. 계수 **1.5**는 관례 '
            '(더 극단만 보려면 3.0).')
    num_cols_iqr = df.select_dtypes('number').columns.tolist()
    if num_cols_iqr:
        ic1, ic2 = st.columns([2, 1])
        ocol = ic1.selectbox('수치형 컬럼', num_cols_iqr, key='iqr_col')
        k = ic2.slider('계수 (1.5=관례, 3.0=극단만)', 1.0, 3.0, 1.5, 0.5, key='iqr_k')
        s = pd.to_numeric(df[ocol], errors='coerce').dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        out = s[(s < low) | (s > high)]
        m = st.columns(4)
        m[0].metric('Q1 / Q3', f'{q1:.1f} / {q3:.1f}')
        m[1].metric('IQR', f'{iqr:.1f}')
        m[2].metric('정상 범위', f'[{low:.1f}, {high:.1f}]')
        m[3].metric('이상치', f'{len(out)}건', f'{len(out)/len(s)*100:.1f}%')
        fig, ax = plt.subplots(figsize=(7, 1.8))
        ax.boxplot(s.values, vert=False, whis=k)
        ax.axvline(low, color='#E53935', ls='--', lw=1)
        ax.axvline(high, color='#E53935', ls='--', lw=1)
        ax.set_title(f'{ocol} — 박스플롯 + Tukey 경계'); st.pyplot(fig)
        insight(
            f"- `{ocol}` 에서 **{len(out)}건({len(out)/len(s)*100:.1f}%)** 이 이상치 경계를 벗어났습니다.\n"
            + ("- 이상치가 많은 편입니다. **무조건 제거하지 마세요** — 긴급정비처럼 "
               "*실제로 의미 있는 극단값*일 수 있습니다. 도메인을 확인하고, 모델이 둔감한 "
               "트리/부스팅을 쓰거나 Robust 스케일링/Log 변환을 고려하세요.\n"
               if len(out) / len(s) > 0.03 else
               "- 이상치 비중이 낮습니다. 제거보다는 그대로 두고 Robust 스케일러로 영향만 줄이는 것을 권장합니다.\n"),
            title='이상치를 어떻게 다룰까', kind='idea', expanded=False)
        show_code(
            "Q1, Q3 = df[col].quantile([0.25, 0.75])\n"
            "IQR = Q3 - Q1\n"
            "low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR\n"
            "outliers = df[(df[col] < low) | (df[col] > high)]")

    # ── 보조 2 · 정보성 결측(MNAR) 분석 ───────────────────────────────
    st.divider()
    st.subheader(':material/rule: 정보성 결측 분석 (결측 자체가 신호인가?)')
    st.caption('어떤 컬럼의 결측이 **무작위(MCAR)** 가 아니라 **특정 상황을 뜻하는 구조적 결측(MNAR)** 인지 '
               '진단합니다. 결측 여부에 따라 타깃이 달라지면 = 정보성 결측 → 함부로 채우거나 버리면 신호가 왜곡됩니다.')
    na_cols = [c for c in df.columns if c != target and df[c].isnull().any()]
    if na_cols and task == 'classification':
        mcol = st.selectbox('결측 분석 대상 컬럼', na_cols, key='mnar_col')
        tnum_m, _ = numeric_target(df[target])
        if tnum_m is not None:
            is_missing = df[mcol].isnull()
            rate_miss = float(tnum_m[is_missing].mean()) if is_missing.any() else float('nan')
            rate_obs = float(tnum_m[~is_missing].mean()) if (~is_missing).any() else float('nan')
            mc = st.columns(3)
            mc[0].metric('결측 비율', f'{is_missing.mean()*100:.1f}%')
            mc[1].metric(f'결측 행의 타깃 양성률', f'{rate_miss*100:.1f}%')
            mc[2].metric(f'비결측 행의 타깃 양성률', f'{rate_obs*100:.1f}%')
            gap = abs(rate_miss - rate_obs) if rate_miss == rate_miss and rate_obs == rate_obs else 0
            if gap > 0.03:
                insight(
                    f"- `{mcol}` 결측 행과 비결측 행의 타깃 양성률이 "
                    f"**{rate_miss*100:.1f}% vs {rate_obs*100:.1f}%** 로 다릅니다 "
                    f"→ **결측 자체가 타깃과 연관 = 정보성 결측(MNAR)**.\n"
                    f"- 권장: 전용 **'N/A' 범주로 명시 인코딩**(`na_category`). 의미가 정확하고 행 손실이 없습니다.\n"
                    f"- ⚠ **단순 삭제/최빈값 대체는 신호를 왜곡**합니다. 특히 최빈값 대체는 "
                    f"'예방정비인데 특정 고장이 났다'는 *거짓 정보*를 주입합니다.\n"
                    f"- ⚠ **행 제거**는 점수가 높아 보여도 함정 — 쉬운 그룹을 통째로 빼고 *다른 모집단*에서 "
                    f"평가하는 것이라 비교 자체가 성립하지 않습니다.",
                    title=f'`{mcol}` 은 정보성 결측 — 채우는 방식이 중요', kind='warn', expanded=True)
            else:
                insight(
                    f"- `{mcol}` 결측/비결측 간 타깃 양성률 차이가 작습니다 "
                    f"({rate_miss*100:.1f}% vs {rate_obs*100:.1f}%) → 결측이 타깃과 크게 연관되진 않습니다.\n"
                    f"- 타입별 표준 대체(수치=중앙값, 범주=최빈값/N/A)로 무난하게 처리할 수 있습니다.",
                    title=f'`{mcol}` 결측 진단', kind='idea', expanded=True)
            show_code(
                "is_missing = df[col].isnull()\n"
                "rate = df.assign(m=is_missing).groupby('m')[TARGET].mean()\n"
                "# 결측/비결측 타깃 비율이 다르면 → 정보성 결측(MNAR) → 'N/A' 범주로 인코딩")
    elif not na_cols:
        st.info('결측이 있는 (타깃 외) 컬럼이 없습니다.')
    else:
        st.info('정보성 결측 분석은 분류 타깃에서 제공됩니다.')


# ══════════════════════════════════════════════════════════════════
# 1. EDA
# ══════════════════════════════════════════════════════════════════
elif page == '1. EDA':
    st.title(':material/query_stats: 실습 1 — EDA')
    df = get_view_df()   # 전처리 계획 적용 토글 반영
    if eda_apply_prep and st.session_state.get('prep_plan'):
        st.success(':material/cleaning_services: 전처리 계획이 적용된 데이터로 탐색 중입니다 '
                   '(결측 대체·범주화 반영). 원본으로 보려면 사이드바 토글을 끄세요.')

    # 알려진 데이터셋이면 가이드의 도메인 해석 포인트를 먼저 제시 (실습가이드 인사이트 반영)
    if profile and profile.get('eda_notes'):
        insight('\n'.join(f'- {n}' for n in profile['eda_notes']),
                title=f"가이드 학습 포인트 — {profile['name']}", kind='idea', expanded=False)

    st.subheader('수치형 통계 요약')
    st.dataframe(df.select_dtypes('number').describe().T, width='stretch')

    # 결측 현황 — 도메인 결측(구조적/정보성)을 바로 보여줌
    na = df.isnull().sum()
    na = na[na > 0].sort_values(ascending=False)
    if len(na):
        st.subheader('결측 현황')
        nat = pd.DataFrame({'결측 수': na, '결측 %': (na / len(df) * 100).round(1)})
        st.dataframe(nat, width='stretch')
        insight(
            f"- 결측이 있는 컬럼 **{len(na)}개**. 결측이 무작위(MCAR)인지, "
            f"특정 상황을 뜻하는 **구조적·정보성 결측(MNAR)**인지 먼저 구분하세요.\n"
            f"- 예: 장비 데이터의 `fault_code`/`fault_category` 결측은 곧 **'예방정비(고장 없음)'**를 뜻합니다 "
            f"— 결측 자체가 신호이므로 함부로 채우거나 버리면 정보가 왜곡됩니다.\n"
            f"- 처리 전략별 비교는 **[2. 전처리 계획]** 탭의 *정보성 결측 분석*에서 직접 실험할 수 있습니다.",
            title='결측을 어떻게 읽을까', kind='idea', expanded=False)

    st.subheader(f'타깃 분포 — {target}')
    vc = df[target].value_counts().sort_index()
    cc1, cc2 = st.columns([1, 2])
    cc1.dataframe(vc.rename('count'))
    ynum, pos = numeric_target(df[target])
    if task == 'classification':
        if ynum is not None and df[target].nunique() == 2:
            cc1.metric(f'양성({pos}) 비율' if pos is not None else '양성 비율',
                       f'{ynum.mean()*100:.1f}%')
        else:
            cc1.metric('클래스 수', f'{df[target].nunique()}')
        fig, ax = plt.subplots(figsize=(5, 3))
        vc.plot(kind='bar', ax=ax); ax.set_title(f'{target} 분포')
        cc2.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(5, 3))
        df[target].plot(kind='hist', bins=40, ax=ax); ax.set_title(f'{target} 분포')
        cc2.pyplot(fig)

    # 타깃 불균형 인사이트 (이진 분류일 때 자동 계산)
    if task == 'classification':
        minor_pct, spw = imbalance_info(df[target])
        if minor_pct is not None:
            if minor_pct < 35:
                insight(
                    f"- 소수 클래스 비율이 **{minor_pct:.1f}%** 로 데이터가 **불균형**합니다 "
                    f"(다수:소수 ≈ **{spw:.1f} : 1**).\n"
                    f"- 불균형에서는 **Accuracy 가 함정**입니다 — 전부 다수 클래스로 찍어도 "
                    f"{100-minor_pct:.1f}% 가 나오기 때문입니다. **PR-AUC·Recall·F1** 를 주지표로 보세요.\n"
                    f"- 모델 단계에서 `class_weight='balanced'` / `scale_pos_weight≈{spw:.1f}` 로 "
                    f"소수 클래스 오분류에 더 큰 페널티를 줄 수 있습니다. "
                    f"**[4. 모델링]** 의 *불균형 보정* 옵션과 *Dummy 베이스라인* 비교로 확인하세요.",
                    title='타깃 불균형 — 평가지표를 바꿔야 한다', kind='warn', expanded=False)
            else:
                insight(f"- 클래스가 비교적 균형적입니다 (소수 클래스 {minor_pct:.1f}%). "
                        f"Accuracy 도 어느 정도 신뢰할 수 있지만, 그래도 F1/ROC-AUC 를 함께 보세요.",
                        title='타깃 균형', kind='good', expanded=False)
    else:
        s = pd.to_numeric(df[target], errors='coerce').dropna()
        if len(s) > 2:
            sk = float(s.skew())
            if abs(sk) > 1:
                insight(f"- 타깃 분포의 **왜도(skew) = {sk:.2f}** 로 한쪽으로 치우쳐 있습니다. "
                        f"로그 변환(`np.log1p`) 후 모델링하면 RMSE/R² 가 개선될 수 있습니다.\n"
                        f"- 회귀 평가는 **RMSE·MAE·R²** 를 함께 보세요.",
                        title='타깃 분포가 치우쳐 있다', kind='idea', expanded=False)

    # 그룹(개체) 구조 — 그룹 누수 경고 (사이드바에서 그룹 컬럼 선택 시)
    if group_col and group_col in df.columns:
        n_groups = df[group_col].nunique()
        rows_per = len(df) / max(n_groups, 1)
        if rows_per >= 1.5:
            insight(
                f"- 그룹 컬럼 **`{group_col}`** 기준으로 개체 **{n_groups}개**가 "
                f"평균 **{rows_per:.1f}행**씩 반복 등장합니다.\n"
                f"- 이때 무작위 `train_test_split` 을 쓰면 **같은 개체가 train·test 양쪽에** 들어가 "
                f"모델이 그 개체를 '외워서' 맞히는 **그룹 누수(group leakage)**가 생기고 성능이 부풀려집니다.\n"
                f"- **[4. 모델링]** 에서 *그룹 누수 방지 분할*을 켜면 `GroupShuffleSplit`·`GroupKFold` 로 "
                f"같은 개체를 한쪽에만 배치합니다.",
                title='그룹 누수에 주의', kind='warn', expanded=False)

    # 도메인 탐색: 범주형 컬럼별 타깃 경향
    st.subheader('범주형 컬럼별 타깃 경향')
    cat_for_group = [c for c in df.select_dtypes(exclude='number').columns
                     if c != target and df[c].nunique() <= 30
                     and not np.issubdtype(df[c].dtype, np.datetime64)]
    if cat_for_group:
        gcol = st.selectbox('그룹 기준 컬럼', cat_for_group)
        if ynum is not None:
            # 수치형/이진 타깃 → 그룹별 평균(=양성률)
            g = ynum.groupby(df[gcol]).mean().sort_values()
            is_rate = (pos is not None) or (df[target].nunique() == 2)
            ylab = f'{target} {"양성률" if is_rate else "평균"}'
            fig, ax = plt.subplots(figsize=(7, 3))
            g.plot(kind='barh', ax=ax); ax.set_xlabel(ylab); ax.set_title(f'{gcol}별 {ylab}')
            st.pyplot(fig)
            # 그룹 간 격차 인사이트 (자동 계산)
            if len(g) >= 2:
                hi, lo = g.index[-1], g.index[0]
                gap = g.iloc[-1] - g.iloc[0]
                unit = '%p' if is_rate else ''
                hv = f'{g.iloc[-1]*100:.1f}%' if is_rate else f'{g.iloc[-1]:.2f}'
                lv = f'{g.iloc[0]*100:.1f}%' if is_rate else f'{g.iloc[0]:.2f}'
                gv = f'{gap*100:.1f}{unit}' if is_rate else f'{gap:.2f}'
                strong = (gap > 0.1) if is_rate else (g.iloc[0] != 0 and gap / abs(g.iloc[0]) > 0.5)
                insight(
                    f"- **`{gcol}` 값에 따라 {ylab}가 갈립니다**: 가장 높은 `{hi}`({hv}) vs "
                    f"가장 낮은 `{lo}`({lv}) — 격차 **{gv}**.\n"
                    + (f"- 격차가 큰 편이라 `{gcol}` 는 **유망한 예측 피처**입니다. "
                       f"높은 그룹이 정비 자원·예비부품 우선 배분 대상이 될 수 있습니다.\n"
                       if strong else
                       f"- 격차가 크지 않아 `{gcol}` 단독으로는 타깃을 잘 가르지 못합니다. "
                       f"다른 변수와의 **조합/상호작용**(트리·부스팅)을 봐야 할 수 있습니다.\n")
                    + "- ⚠ 단, **건수(노출량)** 가 다르면 비율 비교가 불공정할 수 있으니 그룹별 표본 수도 함께 보세요.",
                    title=f'{gcol}별 경향을 어떻게 읽을까', kind='idea', expanded=False)
        else:
            # 다중 범주형 타깃 → 그룹별 클래스 구성비 (정규화 교차표)
            st.caption('다중 범주 타깃: 그룹별 클래스 구성비(행 합=100%)')
            ct = pd.crosstab(df[gcol], df[target], normalize='index')
            st.dataframe((ct * 100).round(1), width='stretch')
            st.bar_chart(ct)
    else:
        st.info('그룹화에 적합한 범주형 컬럼이 없습니다.')

    # ── 수치형 상관행렬 + 타깃 상관 인사이트 (노트북 인사이트 ⑤ 반영) ──
    st.subheader('수치형 변수 상관행렬')
    num_for_corr = df.select_dtypes('number').columns.tolist()
    if len(num_for_corr) >= 2:
        import plotly.express as px
        sel_corr = st.multiselect('포함할 수치형 변수', num_for_corr,
                                  default=num_for_corr[:14], key='eda_corr_sel')
        if len(sel_corr) >= 2:
            corr = df[sel_corr].corr()
            st.plotly_chart(px.imshow(corr, text_auto='.2f', aspect='auto',
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title='수치형 변수 상관행렬'), width='stretch')

            # 타깃과의 상관 — 약한 신호인지 자동 진단
            tnum = pd.to_numeric(df[target], errors='coerce') if target in sel_corr else None
            if target in sel_corr and tnum is not None and tnum.notna().any():
                tcorr = corr[target].drop(target).abs().sort_values(ascending=False)
                top = tcorr.head(3)
                top_txt = ', '.join(f'`{k}`(|r|={v:.2f})' for k, v in top.items())
                max_r = float(tcorr.iloc[0]) if len(tcorr) else 0.0
                if max_r < 0.2:
                    insight(
                        f"- **타깃 `{target}` 과 모든 수치형 변수의 선형 상관이 매우 약합니다** "
                        f"(최대 |r|={max_r:.2f}). 가장 큰 축: {top_txt}.\n"
                        f"- 개별 변수의 선형 신호가 약하면 **로지스틱·선형 모델의 한계**가 분명합니다 "
                        f"→ **트리/부스팅의 비선형·상호작용**이나 **추가 피처 엔지니어링**이 필요합니다.\n"
                        f"- ⚠ **상관계수는 선형 관계만 잡습니다.** r≈0 이 '무관'은 아니며, "
                        f"비선형 효과는 **SHAP/순열 중요도**로 따로 확인하세요 ([4. 모델링] STEP 5).",
                        title='타깃과의 상관이 약하다 — 왜 모델 성능이 안 오를까', kind='idea', expanded=False)
                else:
                    insight(
                        f"- 타깃과 상관이 큰 변수: {top_txt}. 이들이 **핵심 예측 피처** 후보입니다.\n"
                        f"- ⚠ 상관이 매우 높은(예: |r|>0.9) 피처는 **누수**일 수 있습니다 — "
                        f"'예측 시점에 정말 알 수 있는 값인가?'를 점검하세요.",
                        title='타깃과 상관이 큰 변수', kind='idea', expanded=False)

            # 다중공선성 점검 (피처 간 |r|>0.8)
            pair_hi = []
            cm = corr.drop(index=[target], columns=[target], errors='ignore')
            for i in range(len(cm.columns)):
                for j in range(i + 1, len(cm.columns)):
                    r = cm.iloc[i, j]
                    if abs(r) > 0.8:
                        pair_hi.append((cm.columns[i], cm.columns[j], r))
            if pair_hi:
                txt = '\n'.join(f'- `{a}` ↔ `{b}` : r={r:.2f}' for a, b, r in pair_hi[:8])
                insight(
                    f"피처 간 강한 상관(다중공선성) 쌍이 있습니다:\n{txt}\n\n"
                    f"선형 모델에서는 **중복 정보**이므로 한쪽을 정리하면 해석이 안정됩니다. "
                    f"(트리 모델은 상대적으로 둔감)",
                    title='다중공선성 점검', kind='warn', expanded=False)
        else:
            st.info('2개 이상 선택하세요.')

    show_code(
        "print(df.describe())\n"
        "df[TARGET].value_counts().plot(kind='bar')\n"
        "df.groupby('equipment_type')[TARGET].mean().sort_values().plot(kind='barh')\n"
        "import seaborn as sns\n"
        "sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)")

    st.divider()
    st.subheader(':material/analytics: 자동 EDA 리포트 (ydata-profiling)')
    st.caption('생성에 시간이 걸릴 수 있습니다. minimal 모드로 빠르게 생성합니다. '
               '리포트는 한 번 생성하면 페이지를 다시 그려도 유지됩니다.')
    rc1, rc2, rc3 = st.columns([1, 1, 1])
    if rc1.button('리포트 생성 / 보기'):
        with st.spinner('리포트 생성 중...'):
            from ydata_profiling import ProfileReport
            apply_korean_font()   # ydata/seaborn 테마가 폰트를 되돌리기 전 한글 폰트 재적용
            # ydata-profiling 은 plot.font_path 로 플롯 폰트를 지정 → 한글 폰트 파일 직접 전달
            prof = ProfileReport(df, title='EDA Report', minimal=True,
                                 plot={'font_path': KFONT_PATH} if KFONT_PATH else {})
            st.session_state['eda_report_html'] = prof.to_html()

    # 세션에 저장된 리포트는 매 실행마다 다시 렌더 → 내부 탭(Variables 등) 클릭으로
    # rerun 이 일어나도 리포트가 사라지지 않는다.
    report_html = st.session_state.get('eda_report_html')
    if report_html:
        rc2.download_button(
            ':material/download: 리포트 HTML 다운로드',
            report_html.encode('utf-8'), file_name='eda_report.html',
            mime='text/html', key='eda_report_dl',
            help='탭(Variables/Correlations 등) 전환은 다운로드해 브라우저 새 탭에서 열면 가장 안정적으로 동작합니다.')
        if rc3.button(':material/close: 리포트 닫기', key='eda_report_close'):
            del st.session_state['eda_report_html']
            st.rerun()
        import streamlit.components.v1 as components
        components.html(report_html, height=900, scrolling=True)

    show_code(
        "from ydata_profiling import ProfileReport\n"
        "ProfileReport(df, title='EDA Report', minimal=True).to_file('reports/eda_report.html')")

    st.divider()
    st.subheader(':material/compare: sweetviz 리포트 (타깃 기준 분석)')
    st.caption('sweetviz 는 타깃을 지정하면 각 피처가 타깃과 어떻게 연관되는지 자동 비교해 줍니다. '
               '(이진/수치 타깃에서 가장 잘 동작)')
    if st.button('sweetviz 리포트 생성 / 보기', key='sv_btn'):
        with st.spinner('sweetviz 리포트 생성 중...'):
            try:
                import sweetviz as sv
                tgt = target if df[target].nunique() <= 2 else None
                rep = sv.analyze(df, target_feat=tgt)
                html_path = 'reports/sweetviz_report.html'
                os.makedirs('reports', exist_ok=True)
                rep.show_html(html_path, open_browser=False, layout='vertical')
                with open(html_path, encoding='utf-8') as fh:
                    st.session_state['sv_report_html'] = fh.read()
            except Exception as e:
                st.error(f'sweetviz 리포트 생성 실패: {e}')
    sv_html = st.session_state.get('sv_report_html')
    if sv_html:
        st.download_button(':material/download: sweetviz HTML 다운로드',
                           sv_html.encode('utf-8'), file_name='sweetviz_report.html',
                           mime='text/html', key='sv_dl')
        import streamlit.components.v1 as components
        components.html(sv_html, height=900, scrolling=True)
    show_code(
        "import sweetviz as sv\n"
        "report = sv.analyze(df, target_feat='failure_within_30days')\n"
        "report.show_html('reports/sweetviz_report.html', open_browser=False)")


# ══════════════════════════════════════════════════════════════════
# 2. 종합 대시보드 (필터 + KPI + 핵심 차트 동시 모니터링)
# ══════════════════════════════════════════════════════════════════
elif page == '3. 종합 대시보드':
    import plotly.express as px
    st.title(':material/dashboard: 실습 3 — 종합 대시보드 (전처리 결과 최종 확인)')
    st.caption('**용도: 전처리까지 마친 데이터를 한눈에 최종 확인하는 모니터링 보드.** '
               '범주형 필터 → KPI 카드 + 핵심 차트(요약)를 한 화면에서 동시에 봅니다. '
               '차트 종류를 더 보거나 코드 예시가 필요하면 **[6. 시각화 레시피·코드]** 탭을 이용하세요.')
    df = get_view_df()   # 전처리 계획 적용 토글 반영
    if eda_apply_prep and st.session_state.get('prep_plan'):
        st.success(':material/cleaning_services: 전처리 계획이 적용된 데이터로 시각화 중입니다 '
                   '(결측 대체·범주화 반영). 원본으로 보려면 사이드바 토글을 끄세요.')

    num_cols_all = df.select_dtypes('number').columns.tolist()
    cat_cols_all = [c for c in df.select_dtypes(exclude='number').columns
                    if not np.issubdtype(df[c].dtype, np.datetime64) and df[c].nunique() <= 30]

    # 필터
    with st.expander(':material/tune: 필터', expanded=True):
        fcol = st.selectbox('필터 기준 (범주형)', ['(없음)'] + cat_cols_all)
        f = df
        if fcol != '(없음)':
            opts = sorted(df[fcol].dropna().unique().tolist())
            sel = st.multiselect(fcol, opts, default=opts)
            f = df[df[fcol].isin(sel)]

    # KPI
    k = st.columns(4)
    k[0].metric('행 수', f'{len(f):,}')
    f_ynum, f_pos = numeric_target(f[target]) if len(f) else (None, None)
    if f_ynum is not None and df[target].nunique() == 2:
        k[1].metric(f'{target} 양성률', f'{f_ynum.mean()*100:.1f}%')
    elif f_ynum is not None:
        k[1].metric(f'{target} 평균', f'{f_ynum.mean():.2f}')
    else:
        k[1].metric(f'{target} 클래스 수', f'{f[target].nunique()}')
    if num_cols_all:
        mcol = num_cols_all[0]
        k[2].metric(f'{mcol} 평균', f'{f[mcol].mean():.1f}')
    k[3].metric('컬럼 수', f'{df.shape[1]}')

    # 필터에 따라 타깃 지표가 어떻게 움직이는지 — 전체 대비 비교 인사이트
    if fcol != '(없음)' and f_ynum is not None and len(f) and len(f) < len(df):
        all_ynum, _ = numeric_target(df[target])
        if all_ynum is not None:
            base_rate = float(all_ynum.mean())
            filt_rate = float(f_ynum.mean())
            is_rate = df[target].nunique() == 2
            diff = (filt_rate - base_rate)
            unit = '%p' if is_rate else ''
            bv = f'{base_rate*100:.1f}%' if is_rate else f'{base_rate:.2f}'
            fv = f'{filt_rate*100:.1f}%' if is_rate else f'{filt_rate:.2f}'
            dv = f'{diff*100:+.1f}{unit}' if is_rate else f'{diff:+.2f}'
            insight(
                f"- 현재 필터(`{fcol}`)에서 {target} {'양성률' if is_rate else '평균'}은 **{fv}** 로, "
                f"전체 **{bv}** 대비 **{dv}** 입니다.\n"
                f"- 전체보다 높은 구간은 **위험/우선관리 대상**, 낮은 구간은 상대적으로 안정적입니다. "
                f"필터를 바꿔가며 어떤 조건에서 지표가 튀는지 찾아보세요.",
                title='필터에 따른 타깃 변화', kind='idea', expanded=False)

    st.dataframe(f.head(20), width='stretch')

    # 차트 2단
    c1, c2 = st.columns(2)
    if cat_cols_all and num_cols_all:
        with c1:
            xcol = st.selectbox('막대 X (범주형)', cat_cols_all, key='barx')
            ycol = st.selectbox('막대 Y (수치형)', num_cols_all, key='bary')
            g = f.groupby(xcol)[ycol].mean().reset_index()
            st.plotly_chart(px.bar(g, x=xcol, y=ycol, title=f'{xcol}별 {ycol} 평균'),
                            width='stretch')
        with c2:
            bxcol = st.selectbox('박스 Y (수치형)', num_cols_all, key='boxy')
            st.plotly_chart(px.box(f, x=cat_cols_all[0], y=bxcol,
                            title=f'{cat_cols_all[0]}별 {bxcol} 분포'), width='stretch')

    # 시계열 (날짜 컬럼 자동 감지)
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if dt_cols:
        dcol = st.selectbox('시계열 날짜 컬럼', dt_cols)
        ts = f.set_index(dcol).resample('ME').size().reset_index(name='건수')
        st.plotly_chart(px.line(ts, x=dcol, y='건수', title='월별 건수 추이'),
                        width='stretch')

    show_code(
        "import plotly.express as px\n"
        "g = f.groupby('equipment_type')[TARGET].mean().reset_index()\n"
        "st.plotly_chart(px.bar(g, x='equipment_type', y=TARGET))\n"
        "st.plotly_chart(px.box(f, x='equipment_type', y='maintenance_cost'))")


# ══════════════════════════════════════════════════════════════════
# 3. 모델링 · 평가 · 해석
# ══════════════════════════════════════════════════════════════════
elif page == '4. 모델링 · 평가 · 해석':
    from sklearn.model_selection import (train_test_split, cross_val_score,
                                         GroupShuffleSplit, GroupKFold)
    from sklearn.pipeline import Pipeline

    st.title(':material/model_training: 실습 4 — 모델링 · 평가 · 해석')
    X, y, num_cols, cat_cols = get_feature_frame()
    st.caption(f'피처: 수치형 {len(num_cols)} · 범주형 {len(cat_cols)} | 제외: {list(drop_cols) + (leak_known if exclude_leak else [])}')

    # XGBoost 등은 수치 클래스 라벨만 허용 → 비수치 분류 타깃은 라벨 인코딩 (문자열 이진 ROC-AUC 도 해결)
    label_classes = None
    if task == 'classification' and not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        label_classes = le.classes_.tolist()
        st.caption('타깃 라벨 인코딩 → ' + ', '.join(f'{i}={c}' for i, c in enumerate(label_classes)))

    # ── 분할/불균형 옵션 (실습가이드 핵심) ──
    y_bin_numeric = pd.api.types.is_numeric_dtype(y) and y.nunique() == 2
    minor_pct, spw = imbalance_info(y) if task == 'classification' else (None, None)

    opt1, opt2 = st.columns(2)
    # 그룹 누수 방지 분할 — 사이드바에서 그룹 컬럼을 골랐고, X 행에 정렬 가능한 경우
    groups = None
    use_group = False
    if group_col and group_col in df.columns:
        g_all = df.loc[X.index, group_col]
        if g_all.nunique() < len(X):      # 실제로 반복되는 그룹이어야 의미 있음
            use_group = opt1.checkbox(
                f'그룹 누수 방지 분할 (`{group_col}` 기준)', value=True,
                help='같은 개체를 train/test 한쪽에만 배치 — GroupShuffleSplit·GroupKFold')
            if use_group:
                groups = g_all
    balanced = False
    if task == 'classification' and minor_pct is not None and minor_pct < 45:
        balanced = opt2.checkbox(
            f'불균형 보정 (class_weight·scale_pos_weight≈{spw:.1f})', value=True,
            help='소수 클래스 오분류에 더 큰 페널티 — 소수 클래스 Recall 을 끌어올림')

    # ── 분할 ──
    pre = make_preprocessor(num_cols, cat_cols, st.session_state.get('prep_plan'))
    if use_group and groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
        groups_train = groups.iloc[tr_idx]
        st.caption(f':material/groups: 그룹 분할: train {X_train.shape[0]}행 / test {X_test.shape[0]}행 '
                   f'— 같은 `{group_col}` 는 한쪽에만')
    else:
        strat = y if task == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=strat, random_state=42)
        groups_train = None

    all_models = ['Dummy(baseline)', 'Logistic', 'GaussianNB', 'RandomForest', 'GBM', 'XGBoost', 'LightGBM'] \
        if task == 'classification' else ['Linear', 'RandomForest', 'GBM', 'XGBoost', 'LightGBM']
    default_sel = [m for m in ['Dummy(baseline)', 'RandomForest', 'XGBoost', 'LightGBM', 'Logistic', 'Linear']
                   if m in all_models]
    sel = st.multiselect('비교할 모델', all_models, default=default_sel)
    cv = st.slider('교차검증 fold', 3, 10, 5)

    # 평가지표: 불균형 이진 → PR-AUC(average_precision) 주지표. 그 외 분류 → f1_weighted / roc_auc.
    if y_bin_numeric and minor_pct is not None and minor_pct < 45:
        scoring = 'average_precision'   # PR-AUC
    elif y_bin_numeric:
        scoring = 'roc_auc'
    elif task == 'classification':
        scoring = 'f1_weighted'
    else:
        scoring = 'r2'
    metric_label = {'average_precision': 'PR-AUC', 'roc_auc': 'ROC-AUC',
                    'f1_weighted': 'F1(weighted)', 'r2': 'R²'}[scoring]

    if scoring == 'average_precision':
        insight(
            f"- 타깃이 불균형(소수 {minor_pct:.1f}%)이라 비교 주지표를 **PR-AUC(average precision)** 로 둡니다. "
            f"ROC-AUC 는 불균형에서 낙관적으로 보일 수 있습니다.\n"
            f"- **Dummy(baseline)** 와 꼭 비교하세요. Dummy 는 다수 클래스만 찍는 모델 — "
            f"여기보다 못하면 그 모델은 *아무것도 학습하지 못한* 것입니다.\n"
            + (f"- **그룹 누수 방지 분할이 켜져 있습니다** — 같은 `{group_col}` 가 train/test 에 섞이지 않아 "
               f"점수가 무작위 분할보다 낮게(=정직하게) 나오는 게 정상입니다.\n" if use_group else
               f"- 사이드바에서 그룹 컬럼을 고르면 **그룹 누수 방지 분할**을 켤 수 있습니다.\n"),
            title='이 데이터에 맞는 평가 설계', kind='idea', expanded=True)

    # ── STEP 2: 다중 모델 비교 ──
    st.subheader('STEP 2 · 다중 모델 비교')
    if st.button('모델 비교 실행'):
        models = get_models(task, sel, balanced=balanced, spw=(spw or 1.0))
        rows = []
        prog = st.progress(0.0)
        cvk = GroupKFold(n_splits=cv) if groups_train is not None else cv
        for i, (name, model) in enumerate(models.items(), 1):
            pipe = Pipeline([('pre', pre), ('model', model)])
            sc = cross_val_score(pipe, X_train, y_train, cv=cvk, scoring=scoring,
                                 groups=groups_train if groups_train is not None else None)
            rows.append({'Model': name, f'{metric_label}_mean': sc.mean(),
                         f'{metric_label}_std': sc.std()})
            prog.progress(i / len(models))
        rdf = pd.DataFrame(rows).sort_values(f'{metric_label}_mean', ascending=False).reset_index(drop=True)
        st.session_state['result_df'] = rdf
        st.session_state['result_metric'] = metric_label
    if 'result_df' in st.session_state:
        rdf = st.session_state['result_df']
        ml = st.session_state.get('result_metric', metric_label)
        st.dataframe(rdf, width='stretch')
        fig, ax = plt.subplots(figsize=(7, 3))
        rdf.set_index('Model')[f'{ml}_mean'].plot(
            kind='barh', xerr=rdf.set_index('Model')[f'{ml}_std'].values, ax=ax)
        ax.invert_yaxis(); ax.set_title(f'모델별 {ml} 비교')
        st.pyplot(fig)
        best_name = rdf.iloc[0]['Model']
        st.success(f':material/emoji_events: 최고: {best_name} ({ml}={rdf.iloc[0][f"{ml}_mean"]:.3f})')
        # Dummy 대비 인사이트
        if 'Dummy(baseline)' in rdf['Model'].values:
            dummy_v = float(rdf.loc[rdf['Model'] == 'Dummy(baseline)', f'{ml}_mean'].iloc[0])
            best_v = float(rdf.iloc[0][f'{ml}_mean'])
            if best_name != 'Dummy(baseline)' and best_v > dummy_v:
                insight(f"- 최고 모델 **{best_name}**({ml}={best_v:.3f}) 이 "
                        f"**Dummy 베이스라인**({ml}={dummy_v:.3f}) 을 앞섭니다 → 모델이 실제로 신호를 학습했습니다.",
                        title='Dummy 베이스라인 대비', kind='good', expanded=False)
            else:
                insight(f"- ⚠ 최고 모델이 **Dummy 베이스라인을 의미 있게 넘지 못합니다**. "
                        f"피처 신호가 약하거나(상관 분석 참고), 누수 컬럼 제외로 정보가 사라졌을 수 있습니다. "
                        f"피처 엔지니어링이나 임계값 튜닝을 검토하세요.",
                        title='Dummy 베이스라인 대비', kind='warn', expanded=True)

    show_code(
        "from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score\n"
        "from sklearn.dummy import DummyClassifier\n"
        "# 그룹(장비) 기준 분할 — 같은 장비가 train/test 양쪽에 들어가는 누수 방지\n"
        "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n"
        "tr, te = next(gss.split(X, y, groups=df['equipment_id']))\n"
        "# 불균형 → PR-AUC(average_precision)·GroupKFold 로 비교\n"
        "spw = (y_train==0).sum() / (y_train==1).sum()  # scale_pos_weight\n"
        "cross_val_score(pipe, X_train, y_train, cv=GroupKFold(5), groups=g_train,\n"
        "                scoring='average_precision')")

    # ── STEP 3~4: 튜닝 + 최종 학습 ──
    st.subheader('STEP 3~4 · 최종 모델 학습 (XGBoost + Optuna 선택)')
    use_optuna = st.checkbox('Optuna 하이퍼파라미터 튜닝 사용', value=False)
    n_trials = st.slider('Optuna trial 수', 5, 100, 20) if use_optuna else 0
    # 목표 Recall — 예측정비는 '놓친 재고장(FN)'이 가장 치명적 → 임계값을 낮춰 FN 을 줄인다
    target_recall = 0.0
    if y_bin_numeric:
        target_recall = st.slider('목표 Recall (재고장 놓침 최소화용 임계값 튜닝)', 0.5, 0.95, 0.80, 0.05,
                                  help='기본 0.5 임계값 대신, 이 Recall 을 만족하는 임계값을 골라 FN(놓친 양성)을 줄입니다.')
    if st.button('최종 모델 학습'):
        from xgboost import XGBClassifier, XGBRegressor
        Est = XGBClassifier if task == 'classification' else XGBRegressor
        base_kw = dict(random_state=42)
        if task == 'classification':
            base_kw['eval_metric'] = 'logloss'
            if balanced and y_bin_numeric and spw:
                base_kw['scale_pos_weight'] = float(spw)
        cvk = GroupKFold(n_splits=cv) if groups_train is not None else cv
        g_kw = {'groups': groups_train} if groups_train is not None else {}

        best_params = {}
        if use_optuna:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
                m = Pipeline([('pre', pre), ('model', Est(**params, **base_kw))])
                return cross_val_score(m, X_train, y_train, cv=cvk, scoring=scoring, **g_kw).mean()

            st.caption(f'Optuna 베이지안 최적화 — {n_trials} trials · 평가지표 {metric_label}'
                       + (f' · GroupKFold({cv})' if groups_train is not None else f' · KFold({cv})'))
            prog = st.progress(0.0, text='튜닝 시작...')
            log_box = st.empty()          # trial 로그 테이블 (실시간 갱신)
            chart_box = st.empty()        # best-so-far 추이 차트
            trial_rows = []

            def _log_trial(study, trial):
                """각 trial 완료 시 호출 — 단계별 로그/진행바/차트 갱신."""
                val = trial.value if trial.value is not None else float('nan')
                is_best = (study.best_trial.number == trial.number)
                row = {'trial': trial.number,
                       scoring: round(val, 4) if val == val else None,
                       'best_so_far': round(study.best_value, 4),
                       'new_best': bool(is_best)}
                # 하이퍼파라미터도 함께 기록
                for k, v in trial.params.items():
                    row[k] = round(v, 4) if isinstance(v, float) else v
                trial_rows.append(row)
                done = len(trial_rows)
                prog.progress(done / n_trials,
                              text=f'trial {trial.number}  ·  {scoring}={row[scoring]}  ·  '
                                   f'best={study.best_value:.4f}  ({done}/{n_trials})')
                log_df = pd.DataFrame(trial_rows)
                # 최신 trial 이 위로 오도록 역순 표시
                log_box.dataframe(log_df.iloc[::-1], width='stretch', height=300,
                                  hide_index=True)
                chart_box.line_chart(log_df.set_index('trial')[[scoring, 'best_so_far']])

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, callbacks=[_log_trial])
            prog.progress(1.0, text=f'완료 · best {scoring} = {study.best_value:.4f}')
            best_params = study.best_params
            st.success(f'best {scoring} = {study.best_value:.4f}  ·  최적 trial #{study.best_trial.number}')
            st.write('**Best params**', best_params)

        best = Pipeline([('pre', pre), ('model', Est(**best_params, **base_kw))])
        best.fit(X_train, y_train)
        st.session_state['best_model'] = best
        st.session_state['Xy_test'] = (X_test, y_test)
        st.session_state['feature_cols'] = (num_cols, cat_cols, X)

        decision_threshold = 0.5
        y_pred = best.predict(X_test)
        if task == 'classification':
            from sklearn.metrics import (classification_report, confusion_matrix,
                                         roc_auc_score, average_precision_score,
                                         precision_recall_curve)
            if y_bin_numeric:
                y_proba = best.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_proba)
                pr = average_precision_score(y_test, y_proba)
                m1, m2 = st.columns(2)
                m1.metric('Test ROC-AUC', f'{roc:.4f}')
                m2.metric('Test PR-AUC', f'{pr:.4f}')

                # ── 임계값 튜닝 — 목표 Recall 을 만족하는 임계값 선택 ──
                prec, rec, thr = precision_recall_curve(y_test, y_proba)
                cand = np.where(rec[:-1] >= target_recall)[0]
                decision_threshold = float(thr[cand[np.argmax(prec[cand])]]) if len(cand) else 0.5

                st.markdown(f'**임계값 비교 — 기본 0.5 vs 튜닝 {decision_threshold:.3f} '
                            f'(목표 Recall ≥ {target_recall:.2f})**')
                cmp_rows = []
                for label, t in [('기본 0.5', 0.5), (f'튜닝 {decision_threshold:.3f}', decision_threshold)]:
                    yp = (y_proba >= t).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
                    rc = tp / (tp + fn) if (tp + fn) else 0
                    pc = tp / (tp + fp) if (tp + fp) else 0
                    cmp_rows.append({'임계값': label, 'FN(놓침)': int(fn), 'FP(오탐)': int(fp),
                                     'Recall': round(rc, 3), 'Precision': round(pc, 3)})
                cmp_df = pd.DataFrame(cmp_rows)
                st.dataframe(cmp_df, width='stretch', hide_index=True)

                # PR 커브 + 선택 임계값 표시
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(rec, prec, lw=1.5)
                ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve (PR-AUC={pr:.3f})')
                ax.axvline(target_recall, color='#E53935', ls='--', lw=1, label=f'목표 Recall {target_recall:.2f}')
                ax.legend(fontsize=8); st.pyplot(fig)

                fn_base = cmp_df.iloc[0]['FN(놓침)']
                fn_tune = cmp_df.iloc[1]['FN(놓침)']
                insight(
                    f"- 임계값을 0.5 → **{decision_threshold:.3f}** 로 낮추면 **놓친 양성(FN)** 이 "
                    f"{fn_base} → **{fn_tune}건** 으로 줄어듭니다 (Recall↑). 대신 오탐(FP)은 늘어납니다.\n"
                    f"- 예측정비처럼 **놓침의 비용 ≫ 오탐의 비용** 인 문제에서는 이 trade-off 가 합리적입니다.\n"
                    f"- ⚠ 엄밀히는 임계값을 **별도 검증셋**에서 골라야 합니다(여기선 실습 단순화로 test 분포 사용 → 약간 낙관적).\n"
                    f"- 이 임계값은 schema 에 저장되어 **[5. 예측 앱]** 의 판정 기준으로 쓰입니다.",
                    title='임계값을 왜, 어떻게 옮기나', kind='idea', expanded=True)
                yp_final = (y_proba >= decision_threshold).astype(int)
                st.text(classification_report(y_test, yp_final))
                cm = confusion_matrix(y_test, yp_final)
            else:
                st.text(classification_report(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap='Blues')
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, v, ha='center', va='center')
            ax.set_xlabel('예측'); ax.set_ylabel('실제'); ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            st.write(f'**RMSE={rmse:.3f} · MAE={mean_absolute_error(y_test, y_pred):.3f} '
                     f'· R²={r2_score(y_test, y_pred):.3f}**')

        # 모델 저장 (예측 앱에서 사용)
        import joblib
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        joblib.dump(best, f'{ARTIFACT_DIR}/model.pkl')
        joblib.dump({'num_cols': num_cols, 'cat_cols': cat_cols, 'task': task,
                     'cat_options': {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
                     'num_median': {c: float(X[c].median()) for c in num_cols},
                     'num_defaults': {c: float(X[c].median()) for c in num_cols},
                     'decision_threshold': float(decision_threshold),
                     'positive_class': (label_classes[1] if label_classes else 1),
                     'label_classes': label_classes},
                    f'{ARTIFACT_DIR}/schema.pkl')
        st.success(f'모델 저장 완료 → {ARTIFACT_DIR}/model.pkl '
                   + (f'(판정 임계값 {decision_threshold:.3f} 포함) ' if y_bin_numeric else '')
                   + '— [5. 예측 앱]에서 사용')

    show_code(
        "study = optuna.create_study(direction='maximize')\n"
        "study.optimize(objective, n_trials=30)\n"
        "best_model = Pipeline([('pre', preprocessor),\n"
        "    ('model', XGBClassifier(**study.best_params, random_state=42))]).fit(X_train, y_train)\n"
        "print(classification_report(y_test, best_model.predict(X_test)))")

    # ── STEP 5: SHAP ──
    st.subheader('STEP 5 · SHAP 해석')
    if 'best_model' in st.session_state and st.button('SHAP 분석 실행'):
        try:
            import shap
            best = st.session_state['best_model']
            X_test_, _ = st.session_state['Xy_test']
            Xtp = best['pre'].transform(X_test_)
            fn_ = best['pre'].get_feature_names_out()
            if hasattr(Xtp, 'toarray'):
                Xtp = Xtp.toarray()
            Xtp_df = pd.DataFrame(Xtp, columns=fn_)
            expl = shap.TreeExplainer(best['model'])
            sv = expl(Xtp_df)
            mean_abs = np.abs(sv.values).mean(axis=0)
            imp = (pd.DataFrame({'feature': fn_, 'mean_abs_shap': mean_abs})
                   .sort_values('mean_abs_shap', ascending=False).head(15).reset_index(drop=True))
            st.dataframe(imp, width='stretch')
            fig, ax = plt.subplots(figsize=(7, 4))
            imp.set_index('feature')['mean_abs_shap'].iloc[::-1].plot(kind='barh', ax=ax)
            ax.set_title('평균 |SHAP| 상위 피처'); st.pyplot(fig)

            # SHAP 결과 보관 → 아래 단일 샘플 waterfall 에서 재사용
            st.session_state['shap_pack'] = {'values': sv, 'Xdf': Xtp_df, 'top': imp}

            # 누수 피처가 상위에 있으면 경고
            leak_in_top = [c for c in (leak_known if profile else [])
                           if any(c in str(f) for f in imp['feature'].head(8))]
            insight(
                f"- 상위 피처가 모델의 판단 근거입니다: " +
                ', '.join(f'`{f}`' for f in imp['feature'].head(5)) + ".\n"
                + (f"- ⚠ 상위에 **누수 의심(사후 정보) 변수**({', '.join(leak_in_top)})가 있습니다. "
                   f"'고장이 난 뒤에야 알 수 있는 정보'에 모델이 기대고 있다는 신호 — 사이드바에서 "
                   f"*누수 의심 컬럼 제외*를 켜고 다시 학습해 중요도가 어떻게 바뀌는지 비교하세요.\n"
                   if leak_in_top else
                   "- ⚠ 상관계수로는 약했던 변수가 SHAP 상위에 올 수 있습니다 — 비선형·상호작용 효과 때문입니다.\n"),
                title='SHAP 중요도를 어떻게 읽을까', kind='idea', expanded=True)
        except Exception as e:
            st.error(f'SHAP 분석 불가 (트리 기반 모델에서만 동작): {e}')

    # 단일 샘플 예측 근거 (waterfall) — 직전 SHAP 분석 결과 재사용
    if 'shap_pack' in st.session_state:
        st.markdown('**단일 샘플 예측 근거 (waterfall)** — 한 건의 예측을 어떤 피처가 밀고/당겼는지')
        pack = st.session_state['shap_pack']
        n_rows = pack['values'].values.shape[0]
        idx = st.number_input('test 샘플 인덱스', 0, max(n_rows - 1, 0), 0, key='shap_wf_idx')
        try:
            import shap
            import matplotlib.pyplot as _plt
            _plt.figure()
            shap.plots.waterfall(pack['values'][int(idx)], show=False)
            st.pyplot(_plt.gcf())
            insight(
                "- 막대가 **오른쪽(양수)** 이면 그 피처가 예측(양성 확률)을 **올렸고**, "
                "**왼쪽(음수)** 이면 **내렸습니다**. 누적 합이 이 샘플의 최종 예측을 만듭니다.\n"
                "- 같은 피처라도 샘플마다 기여 방향/크기가 달라집니다 — 그래서 SHAP 은 '왜 이 건이 이렇게 예측됐나'를 설명합니다.",
                title='waterfall 읽는 법', kind='idea', expanded=False)
        except Exception as e:
            st.caption(f'waterfall 표시 불가: {e}')

    show_code(
        "explainer = shap.TreeExplainer(best_model['model'])\n"
        "shap_values = explainer(X_test_pre_df)\n"
        "shap.summary_plot(shap_values, X_test_pre_df)   # 전체 중요도\n"
        "shap.plots.waterfall(shap_values[0])            # 단일 샘플 근거")


# ══════════════════════════════════════════════════════════════════
# 4. 예측 앱
# ══════════════════════════════════════════════════════════════════
elif page == '5. 예측 앱':
    import joblib
    st.title(':material/online_prediction: 실습 5 — 예측 앱')

    model_path = f'{ARTIFACT_DIR}/model.pkl'
    schema_path = f'{ARTIFACT_DIR}/schema.pkl'
    if not (os.path.exists(model_path) and os.path.exists(schema_path)):
        st.warning('학습된 모델이 없습니다. 먼저 [4. 모델링 · 평가 · 해석] 페이지에서 "최종 모델 학습"을 실행하세요.')
        st.stop()

    model = joblib.load(model_path)
    schema = joblib.load(schema_path)
    # 구버전 schema 호환 (키 누락 방어)
    cat_options = schema.get('cat_options', {})
    num_median = schema.get('num_median', schema.get('num_defaults', {}))
    schema_task = schema.get('task', task)
    THR = float(schema.get('decision_threshold', 0.5))   # 학습 때 튜닝한 판정 임계값

    classes = schema.get('label_classes')
    feat_cat = [c for c in schema.get('cat_cols', [])]
    feat_num = [c for c in schema.get('num_cols', [])]

    # 모델이 이진 확률을 내는지 확인 → 임계값 판정 사용 가능 여부
    is_binary_proba = (schema_task == 'classification'
                       and hasattr(model, 'predict_proba')
                       and len(getattr(model, 'classes_', [0, 1])) == 2)
    if is_binary_proba and abs(THR - 0.5) > 1e-6:
        st.caption(f':material/tune: 판정 임계값 **{THR:.3f}** 적용 중 (모델링에서 목표 Recall 로 튜닝됨). '
                   f'양성 확률이 이 값 이상이면 양성으로 판정합니다.')

    def _cat_opts(col):
        base = cat_options.get(col) or (sorted(df[col].dropna().unique().tolist()) if col in df.columns else ['N/A'])
        return list(base)

    # 입력 위젯 기본값(세션) 초기화 — 랜덤 채우기로 덮어쓸 수 있도록 key 사용
    for col in feat_cat:
        k = f'pin_{col}'
        if k not in st.session_state:
            o = _cat_opts(col)
            st.session_state[k] = o[0] if o else 'N/A'
    for col in feat_num:
        k = f'pin_{col}'
        if k not in st.session_state:
            st.session_state[k] = float(num_median.get(col, 0.0))

    # ── 전체 데이터에서 랜덤 1건 불러오기 ──
    st.subheader('입력값')
    cR1, cR2 = st.columns([1, 2])
    if cR1.button(':material/casino: 랜덤 샘플 불러오기'):
        row = df.sample(1).iloc[0]
        for col in feat_cat + feat_num:
            if col in df.columns:
                v = row[col]
                if col in feat_num:
                    st.session_state[f'pin_{col}'] = float(v) if pd.notna(v) else float(num_median.get(col, 0.0))
                else:
                    st.session_state[f'pin_{col}'] = 'N/A' if pd.isna(v) else v
        st.session_state['rand_actual'] = None if target not in row else row[target]
        st.rerun()
    if st.session_state.get('rand_actual') is not None:
        cR2.info(f'불러온 샘플의 실제 {target} = **{st.session_state["rand_actual"]}**')

    # ── 입력 위젯 (session_state 키 기반) ──
    inp = {}
    c1, c2 = st.columns(2)
    for i, col in enumerate(feat_cat):
        with (c1 if i % 2 == 0 else c2):
            opts = _cat_opts(col)
            cur = st.session_state.get(f'pin_{col}')
            if cur is not None and cur not in opts:
                opts.append(cur)
            inp[col] = st.selectbox(col, opts, key=f'pin_{col}')
    for i, col in enumerate(feat_num):
        with (c1 if i % 2 == 0 else c2):
            inp[col] = st.number_input(col, key=f'pin_{col}')

    if st.button('예측하기', type='primary'):
        x = pd.DataFrame([inp])
        if is_binary_proba:
            # 양성 클래스 확률 → 저장된 임계값으로 판정 (기본 0.5 가 아닐 수 있음)
            pos_idx = list(model.classes_).index(model.classes_[1])
            proba_pos = float(model.predict_proba(x)[0, pos_idx])
            pred = int(proba_pos >= THR)
            pred_disp = classes[pred] if classes is not None else pred
            st.metric('예측 결과', str(pred_disp),
                      f'양성 확률 {proba_pos*100:.1f}% (임계값 {THR:.2f})')
            if pred == 1:
                st.error(f'⚠ 양성 판정 — 양성 확률 {proba_pos*100:.1f}% ≥ 임계값 {THR:.2f}')
            else:
                st.success(f'✅ 음성 판정 — 양성 확률 {proba_pos*100:.1f}% < 임계값 {THR:.2f}')
        else:
            pred = model.predict(x)[0]
            pred_disp = classes[int(pred)] if classes is not None else pred
            if schema_task == 'classification':
                try:
                    proba = model.predict_proba(x)[0]
                    st.metric('예측 결과', str(pred_disp), f'확신도 {float(np.max(proba))*100:.1f}%')
                except Exception:
                    st.metric('예측 결과', str(pred_disp))
            else:
                st.metric('예측 결과', f'{pred:.3f}')
        # 랜덤 샘플이면 실제값과 일치 여부 표시
        act = st.session_state.get('rand_actual')
        if act is not None:
            ok = str(act) == str(pred_disp)
            (st.success if ok else st.error)(
                f'실제 {target} = {act} · 예측 = {pred_disp} → {"적중" if ok else "불일치"}')

        log = x.copy()
        log['prediction'] = pred_disp
        log['timestamp'] = datetime.datetime.now()
        log.to_csv(PRED_LOG, mode='a', header=not os.path.exists(PRED_LOG), index=False)
        st.info(f'결과가 {PRED_LOG} 에 저장되었습니다.')

    # ── 전체 데이터 랜덤 배치 테스트 ──
    st.divider()
    st.subheader(':material/science: 전체 데이터 랜덤 배치 테스트')
    st.caption('전체 데이터에서 무작위로 N건을 추출해 모델 예측과 실제 타깃을 한 번에 비교합니다.')
    bc1, bc2 = st.columns([1, 2])
    n = bc1.slider('샘플 수', 5, int(min(500, len(df))), min(50, len(df)))
    if bc2.button(':material/play_arrow: 랜덤 배치 테스트 실행'):
        avail = [c for c in feat_num + feat_cat if c in df.columns]
        sample = df.sample(min(n, len(df)))
        # 결측대체·범주화·스케일링은 model 파이프라인 안에서 처리 → 원본 그대로 입력
        Xs = sample[avail].copy()
        if is_binary_proba:
            proba_pos = model.predict_proba(Xs)[:, 1]
            preds = (proba_pos >= THR).astype(int)
        else:
            preds = model.predict(Xs)
        preds_disp = [classes[int(p)] for p in preds] if classes is not None else list(preds)

        res = sample[avail].copy()
        if is_binary_proba:
            res.insert(0, '양성확률', np.round(proba_pos, 3))
        res.insert(0, '예측', preds_disp)
        if target in sample.columns:
            actual = sample[target].values
            res.insert(0, '실제', actual)
            if schema_task == 'classification':
                correct = pd.Series(preds_disp).astype(str).values == pd.Series(actual).astype(str).values
                res.insert(0, '정답', correct)
                m1, m2, m3 = st.columns(3)
                m1.metric('정확도 (Accuracy)', f'{correct.mean()*100:.1f}%',
                          f'{int(correct.sum())} / {len(correct)} 적중')
                if is_binary_proba:
                    av = pd.to_numeric(pd.Series(actual), errors='coerce')
                    tp = int(((preds == 1) & (av == 1)).sum())
                    fn = int(((preds == 0) & (av == 1)).sum())
                    recall = tp / (tp + fn) if (tp + fn) else float('nan')
                    m2.metric('재현율 Recall (양성 포착)', f'{recall*100:.1f}%' if recall == recall else 'N/A',
                              f'놓침 FN={fn}')
                m3.metric('샘플 수', f'{len(correct)}')
            else:
                err = actual - np.asarray(preds, dtype=float)
                res.insert(0, '오차', err.round(3))
                rmse = float((err ** 2).mean() ** 0.5)
                mae = float(np.abs(err).mean())
                m1, m2 = st.columns(2)
                m1.metric('RMSE', f'{rmse:.3f}')
                m2.metric('MAE', f'{mae:.3f}')
        st.dataframe(res.reset_index(drop=True), width='stretch', height=420)

    if os.path.exists(PRED_LOG):
        st.subheader('예측 히스토리')
        st.dataframe(pd.read_csv(PRED_LOG).tail(20), width='stretch')

    show_code(
        "# 전체 데이터에서 랜덤 추출 → 예측 vs 실제 비교\n"
        "sample = df.sample(n)\n"
        "preds = model.predict(sample[feature_cols])\n"
        "accuracy = (preds == sample[TARGET]).mean()")


# ══════════════════════════════════════════════════════════════════
# 5. 시각화 레시피·코드 (차트 카탈로그 + show_code, eda_visualization_guide.docx 기반)
# ══════════════════════════════════════════════════════════════════
elif page == '6. 시각화 레시피·코드':
    import plotly.express as px
    st.title(':material/auto_graph: 시각화 레시피 · 코드')
    st.caption('**용도: 차트 종류별 레퍼런스 + 코드 학습.** 12종 차트를 하나씩 골라 그려보고, '
               '각 차트의 `show_code`로 구현 코드를 확인합니다. 필터링해서 여러 지표를 한 화면에서 '
               '모니터링하려면 **[3. 종합 대시보드]** 탭을 이용하세요.')

    num_cols_all = df.select_dtypes('number').columns.tolist()
    cat_cols_all = [c for c in df.select_dtypes(exclude='number').columns
                    if not np.issubdtype(df[c].dtype, np.datetime64)]
    dt_cols_all = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

    # ── 데이터셋 자동 감지 + 가이드 해석 포인트 (dataset_profiles 공유) ──
    if profile:
        st.success(f":material/verified: 감지된 데이터셋: **{profile['name']}**  ·  {profile['objective']}")
        with st.expander(':material/lightbulb: 가이드 해석 포인트'):
            for n in profile['eda_notes']:
                st.markdown(f'- {n}')
    else:
        st.caption('범용 모드 — 업로드한 데이터의 컬럼 타입을 자동 감지해 시각화합니다.')

    CHARTS = [
        '1. 범주형 비율 (파이/도넛)',
        '2. 범주형 빈도 (막대)',
        '3. 수치형 분포 (히스토그램)',
        '4. 그룹별 분포 (박스/바이올린)',
        '5. 시계열 추이 (라인)',
        '6. 시간대 패턴 (hour별 평균)',
        '7. 이변량 산점도',
        '8. 상관행렬 히트맵',
        '9. 타깃/이상 비율 (불균형)',
        '10. 그룹별 KPI 분포 비교 (정상 vs 이상)',
        '11. 자기상관 ACF/PACF',
        '12. 페어플롯 (산점도 행렬)',
    ]
    choice = st.selectbox('시각화 항목', CHARTS)

    # ── 1. 범주형 비율 (파이/도넛) ──
    if choice.startswith('1.'):
        col = st.selectbox('범주형 컬럼', cat_cols_all or df.columns.tolist())
        donut = st.checkbox('도넛형', value=True)
        vc = df[col].value_counts()
        fig = px.pie(values=vc.values, names=vc.index, hole=0.45 if donut else 0,
                     title=f'{col} 비율')
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, width='stretch')
        show_code("df['maintenance_type'].value_counts().plot.pie(autopct='%1.1f%%')")

    # ── 2. 범주형 빈도 (막대) ──
    elif choice.startswith('2.'):
        col = st.selectbox('범주형 컬럼', cat_cols_all or df.columns.tolist())
        excl = st.text_input('제외할 값(쉼표 구분, 예: 예방정비)', '')
        d = df
        if excl.strip():
            d = df[~df[col].astype(str).isin([x.strip() for x in excl.split(',')])]
        vc = d[col].value_counts()
        st.plotly_chart(px.bar(x=vc.index, y=vc.values,
                        labels={'x': col, 'y': '빈도'}, title=f'{col} 빈도'), width='stretch')
        show_code("df['equipment_type'].value_counts().plot.bar()")

    # ── 3. 수치형 분포 (히스토그램) ──
    elif choice.startswith('3.'):
        col = st.selectbox('수치형 컬럼', num_cols_all)
        bins = st.slider('구간 수', 10, 100, 40)
        hue = st.selectbox('색상 구분(선택)', ['(없음)'] + cat_cols_all + num_cols_all)
        kw = {} if hue == '(없음)' else {'color': df[hue].astype(str)}
        st.plotly_chart(px.histogram(df, x=col, nbins=bins, title=f'{col} 분포', **kw),
                        width='stretch')
        show_code("axes.hist(df['severity'], bins=40)")

    # ── 4. 그룹별 분포 (박스/바이올린) ──
    elif choice.startswith('4.'):
        x = st.selectbox('그룹 (범주형)', cat_cols_all or num_cols_all)
        y = st.selectbox('값 (수치형)', num_cols_all)
        kind = st.radio('차트', ['박스플롯', '바이올린'], horizontal=True)
        fn = px.box if kind == '박스플롯' else px.violin
        extra = {'box': True} if kind == '바이올린' else {}
        st.plotly_chart(fn(df, x=x, y=y, color=x, title=f'{x}별 {y} 분포', **extra),
                        width='stretch')
        show_code("sns.boxplot(data=df, x='maintenance_type', y='downtime_hours')")

    # ── 5. 시계열 추이 (라인) ──
    elif choice.startswith('5.'):
        if not dt_cols_all:
            st.info('날짜/시간 컬럼이 없습니다.')
        else:
            dcol = st.selectbox('날짜 컬럼', dt_cols_all)
            y = st.selectbox('값 (수치형)', num_cols_all)
            freq = st.selectbox('집계 주기', ['일(D)', '주(W)', '월(ME)', '시간(h)'], index=2)
            fmap = {'일(D)': 'D', '주(W)': 'W', '월(ME)': 'ME', '시간(h)': 'h'}
            gcol = st.selectbox('시리즈 구분(선택)', ['(없음)'] + cat_cols_all)
            d = df[[dcol, y] + ([gcol] if gcol != '(없음)' else [])].dropna(subset=[dcol])
            if gcol == '(없음)':
                ts = d.set_index(dcol)[y].resample(fmap[freq]).mean().reset_index()
                fig = px.line(ts, x=dcol, y=y, title=f'{y} 추이 ({freq})')
            else:
                top = d[gcol].value_counts().head(6).index
                ts = (d[d[gcol].isin(top)].set_index(dcol).groupby(gcol)[y]
                      .resample(fmap[freq]).mean().reset_index())
                fig = px.line(ts, x=dcol, y=y, color=gcol, title=f'{y} 추이 ({freq})')
            st.plotly_chart(fig, width='stretch')
        show_code("df.set_index('observation_datetime')['temperature'].resample('ME').mean().plot()")

    # ── 6. 시간대 패턴 (hour별 평균) ──
    elif choice.startswith('6.'):
        hour_col = 'hour' if 'hour' in df.columns else None
        d = df.copy()
        if hour_col is None and dt_cols_all:
            dcol = st.selectbox('시각 추출용 날짜 컬럼', dt_cols_all)
            d['hour'] = d[dcol].dt.hour
            hour_col = 'hour'
        if hour_col is None:
            st.info("'hour' 컬럼이나 날짜 컬럼이 필요합니다.")
        else:
            y = st.selectbox('값 (수치형)', [c for c in num_cols_all if c != 'hour'])
            gcol = st.selectbox('지역/그룹 구분(선택)', ['(없음)'] + cat_cols_all)
            if gcol == '(없음)':
                g = d.groupby(hour_col)[y].mean().reset_index()
                fig = px.line(g, x=hour_col, y=y, markers=True, title=f'시간대별 {y} 평균')
            else:
                g = d.groupby([gcol, hour_col])[y].mean().reset_index()
                fig = px.line(g, x=hour_col, y=y, color=gcol, markers=True,
                              title=f'지역별 시간대 {y} 패턴')
            st.plotly_chart(fig, width='stretch')
        show_code("df.groupby(['station_region','hour'])['temperature'].mean()")

    # ── 7. 이변량 산점도 ──
    elif choice.startswith('7.'):
        x = st.selectbox('X (수치형)', num_cols_all, index=0)
        y = st.selectbox('Y (수치형)', num_cols_all, index=min(1, len(num_cols_all) - 1))
        hue = st.selectbox('색상 구분(선택)', ['(없음)'] + cat_cols_all + num_cols_all)
        kw = {} if hue == '(없음)' else {'color': df[hue].astype(str)}
        samp = df.sample(min(3000, len(df)))   # 과밀 방지 샘플링
        st.plotly_chart(px.scatter(samp, x=x, y=y, opacity=0.5,
                        title=f'{x} vs {y}', **({'color': samp[hue].astype(str)} if hue != '(없음)' else {})),
                        width='stretch')
        if len(df) > 3000:
            st.caption(f'과밀 방지로 {len(df)}건 중 3,000건 샘플 표시')
        show_code("ax.scatter(df['equipment_age_years'], df['cumulative_operating_hours'])")

    # ── 8. 상관행렬 히트맵 ──
    elif choice.startswith('8.'):
        sel = st.multiselect('수치형 변수', num_cols_all, default=num_cols_all[:12])
        if len(sel) >= 2:
            corr = df[sel].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title='수치형 변수 상관행렬')
            st.plotly_chart(fig, width='stretch')
        else:
            st.info('2개 이상 선택하세요.')
        show_code("sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='RdBu_r')")

    # ── 9. 타깃/이상 비율 (불균형) ──
    elif choice.startswith('9.'):
        bin_cols = [c for c in df.columns if df[c].nunique() == 2]
        tcol = st.selectbox('타깃/이상 컬럼 (이진)', bin_cols or df.columns.tolist())
        vc = df[tcol].value_counts().sort_index()
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(px.bar(x=vc.index.astype(str), y=vc.values,
                        labels={'x': tcol, 'y': '건수'}, title=f'{tcol} 분포'), width='stretch')
        minor = vc.min() / vc.sum() * 100
        c2.metric('소수 클래스 비율', f'{minor:.1f}%')
        c2.caption('불균형이 심하면 Accuracy 대신 F1/ROC-AUC/PR-AUC 사용')
        insight(
            f"- 소수 클래스 비율 **{minor:.1f}%**. "
            + ("불균형이 큽니다 → 전부 다수로 찍어도 정확도가 높게 나오는 **Accuracy 함정**에 주의. "
               "**PR-AUC·Recall** 중심으로 평가하고, `class_weight`/`scale_pos_weight` 또는 SMOTE 를 검토하세요.\n"
               if minor < 35 else "비교적 균형적이라 Accuracy 도 참고할 수 있습니다.\n"),
            title='불균형 진단', kind='warn' if minor < 35 else 'good', expanded=False)
        show_code("df['failure_within_30days'].value_counts()")

    # ── 10. 그룹별 KPI 분포 비교 (정상 vs 이상) ──
    elif choice.startswith('10.'):
        bin_cols = [c for c in df.columns if df[c].nunique() == 2]
        gcol = st.selectbox('비교 기준 (이진, 예: is_anomaly/failure_within_30days)',
                            bin_cols or cat_cols_all)
        metrics = st.multiselect('비교할 수치형 KPI', num_cols_all,
                                 default=num_cols_all[:6])
        if metrics:
            ncol = 3
            nrow = int(np.ceil(len(metrics) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
            axes = np.atleast_1d(axes).ravel()
            for i, m in enumerate(metrics):
                for v in sorted(df[gcol].dropna().unique()):
                    axes[i].hist(df[df[gcol] == v][m].dropna(), bins=30, alpha=0.55,
                                 label=f'{gcol}={v}')
                axes[i].set_title(m, fontsize=10); axes[i].legend(fontsize=7)
            for j in range(len(metrics), len(axes)):
                axes[j].axis('off')
            plt.tight_layout(); st.pyplot(fig)
        show_code("for label in [0,1]:\n    ax.hist(df[df['is_anomaly']==label][metric], alpha=0.6)")

    # ── 11. 자기상관 ACF/PACF ──
    elif choice.startswith('11.'):
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        y = st.selectbox('시계열 수치형', num_cols_all)
        gcol = st.selectbox('단일 시리즈 필터(선택)', ['(전체)'] + cat_cols_all)
        series = df[y].dropna()
        if gcol != '(전체)':
            v = st.selectbox(f'{gcol} 값', sorted(df[gcol].dropna().unique()))
            series = df[df[gcol] == v][y].dropna()
        lags = st.slider('lags', 10, 96, 48)
        if len(series) > lags + 1:
            fig, axes = plt.subplots(2, 1, figsize=(11, 6))
            plot_acf(series.values, lags=lags, ax=axes[0], title=f'ACF - {y}')
            plot_pacf(series.values, lags=lags, ax=axes[1], method='ywm', title=f'PACF - {y}')
            plt.tight_layout(); st.pyplot(fig)
            st.caption('24 등 특정 lag에서 피크 → 주기성(계절성). SARIMA seasonal order 근거')
        else:
            st.info('데이터가 lags 보다 적습니다. lags 를 줄이세요.')
        show_code("from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n"
                  "plot_acf(series, lags=48); plot_pacf(series, lags=48, method='ywm')")

    # ── 12. 페어플롯 ──
    elif choice.startswith('12.'):
        import seaborn as sns
        sel = st.multiselect('변수 (3~5개 권장)', num_cols_all, default=num_cols_all[:4])
        hue = st.selectbox('색상 구분(선택)', ['(없음)'] + [c for c in df.columns if df[c].nunique() <= 6])
        if 2 <= len(sel) <= 6:
            cols = sel + ([hue] if hue != '(없음)' else [])
            samp = df[cols].dropna().sample(min(1500, len(df)))
            with st.spinner('페어플롯 생성 중...'):
                g = sns.pairplot(samp, vars=sel, corner=True,
                                 hue=None if hue == '(없음)' else hue,
                                 plot_kws={'alpha': 0.4, 's': 15})
                st.pyplot(g.figure)
        else:
            st.info('2~6개 변수를 선택하세요.')
        show_code("sns.pairplot(df[key_vars], hue='is_anomaly', diag_kind='hist')")
