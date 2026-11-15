import pandas as pd
import joblib
from prefect import flow, task
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = 'data/dataset1_equipment_maintenance.xlsx'
TARGET = 'failure_within_30days'

@task
def load_data(path):
    df = pd.read_excel(path, sheet_name='데이터')
    df['maintenance_date'] = pd.to_datetime(df['maintenance_date'])
    return df

@task
def preprocess(df):
    df = df.copy()
    df['fault_code'] = df['fault_code'].fillna('N/A')
    df['fault_category'] = df['fault_category'].fillna('N/A')
    df['days_since_last_maintenance'] = df['days_since_last_maintenance'].fillna(0)
    # maintenance_date 만 제거. equipment_id 는 그룹 분할용으로 유지 후 분할 직전 제거.
    return df.drop(columns=['maintenance_date'])

@task
def train_model(df):
    groups = df['equipment_id']
    X = df.drop(columns=[TARGET, 'equipment_id']); y = df[TARGET]
    # 그룹(장비) 기준 분할 — 같은 장비가 train/test에 동시에 들어가는 누수 방지
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    # 불균형 보정
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    num = X.select_dtypes('number').columns.tolist()
    cat = X.select_dtypes('object').columns.tolist()
    pre = ColumnTransformer([('num', StandardScaler(), num),
                             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])
    pipe = Pipeline([('pre', pre),
                     ('model', XGBClassifier(random_state=42, eval_metric='logloss',
                                             scale_pos_weight=spw))])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    ap = average_precision_score(y_te, proba)
    print(f'Test ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}')
    joblib.dump(pipe, 'artifacts/model.pkl')
    return ap

@flow(name='equipment-failure-pipeline')
def ml_pipeline(data_path=DATA_PATH):
    df = load_data(data_path)
    df = preprocess(df)
    return train_model(df)

if __name__ == '__main__':
    ml_pipeline()
