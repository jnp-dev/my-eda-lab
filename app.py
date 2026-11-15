import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='장비 정비 EDA Dashboard', layout='wide')
st.title('📊 장비 정비이력 분석 대시보드')

DATA_PATH = 'data/dataset1_equipment_maintenance.xlsx'
TARGET = 'failure_within_30days'

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name='데이터')
    df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    return df

df = load_data()

# ── 사이드바 필터 ──
with st.sidebar:
    st.header('필터')
    types = st.multiselect('장비 유형', sorted(df['equipment_type'].unique()),
                           default=sorted(df['equipment_type'].unique()))
    sev = st.slider('고장 심각도(severity) 범위', 0, 5, (0, 5))

f = df[df['equipment_type'].isin(types) &
       df['severity'].between(*sev)]

# ── KPI 카드 ──
k1, k2, k3, k4 = st.columns(4)
k1.metric('정비 건수', f'{len(f):,}')
k2.metric('30일내 재고장률', f"{f[TARGET].mean()*100:.1f}%")
k3.metric('평균 정비비용(만원)', f"{f['maintenance_cost'].mean():.0f}")
k4.metric('평균 가동중단(h)', f"{f['downtime_hours'].mean():.1f}")

st.dataframe(f.head(20), use_container_width=True)

# ── 시각화 (2단 레이아웃) ──
c1, c2 = st.columns(2)
with c1:
    g = f.groupby('equipment_type')[TARGET].mean().reset_index()
    st.plotly_chart(px.bar(g, x='equipment_type', y=TARGET,
                           title='장비유형별 재고장률'), use_container_width=True)
with c2:
    st.plotly_chart(px.box(f, x='equipment_type', y='maintenance_cost',
                           title='장비유형별 정비비용 분포'), use_container_width=True)

# ── 월별 정비 추이 ──
ts = (f.set_index('maintenance_date').resample('M').size().reset_index(name='건수'))
st.plotly_chart(px.line(ts, x='maintenance_date', y='건수',
                        title='월별 정비 건수 추이'), use_container_width=True)
