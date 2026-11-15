import os, datetime
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='장비 재고장 예측', layout='centered')
st.title('🔮 장비 30일내 재고장 예측 앱')

model = joblib.load('artifacts/model.pkl')
schema = joblib.load('artifacts/schema.pkl')
THR = schema.get('decision_threshold', 0.5)   # 학습 때 정한 임계값(기본 0.5 아님)

st.subheader('장비/정비 정보 입력')
inp = {}
# 범주형 → selectbox
for c in schema['cat_cols']:
    inp[c] = st.selectbox(c, schema['cat_options'][c])
# 수치형 → number_input (기본값=학습 데이터 중앙값 → 부분 입력 시에도 합리적)
for c in schema['num_cols']:
    inp[c] = st.number_input(c, value=float(schema['num_defaults'][c]))

if st.button('예측하기'):
    input_df = pd.DataFrame([inp])
    proba = float(model.predict_proba(input_df)[0, 1])
    pred = int(proba >= THR)                   # 튜닝된 임계값으로 판정
    if pred == 1:
        st.error(f'⚠ 30일 내 재고장 위험 — 확률 {proba*100:.1f}% (임계값 {THR:.2f})')
    else:
        st.success(f'✅ 재고장 위험 낮음 — 재고장 확률 {proba*100:.1f}% (임계값 {THR:.2f})')

    # 결과를 CSV에 자동 누적 저장
    log = input_df.copy()
    log['prediction'] = pred
    log['proba'] = round(proba, 4)
    log['timestamp'] = datetime.datetime.now()
    path = 'predictions.csv'
    log.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    st.info('결과가 predictions.csv에 저장되었습니다.')
