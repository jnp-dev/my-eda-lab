# -*- coding: utf-8 -*-
"""
Orange3 워크플로우(.ows) 생성기 — 데이터분석_실습지.md 의 실습 순서를 그대로 따른다.

.ows 는 XML 이고, 위젯 설정은 base64(pickle) 로 들어간다. 손으로 쓰기 어렵기 때문에
Orange 가 들어 있는 파이썬으로 이 스크립트를 돌려 생성한다.

실행
  /Applications/Orange.app/Contents/MacOS/python workflow/tools/build_workflows.py
"""
from __future__ import annotations

import importlib
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from orangecanvas.scheme.readwrite import dumps
from orangewidget.utils.signals import Input, Output
from Orange.widgets.utils.filedialogs import RecentPath

WF_DIR = Path(__file__).resolve().parents[1]          # workflow/
DATA_TAB = "data/equipment_maintenance.tab"
NEW_TAB = "data/equipment_new_input.tab"

FONT = ("Helvetica", 13)


# ─────────────────────────────────────────────────────────────────────────────
# 위젯 메타 조회
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: dict[str, type] = {}


def widget_class(qname: str) -> type:
    if qname not in _CACHE:
        mod, cls = qname.rsplit(".", 1)
        _CACHE[qname] = getattr(importlib.import_module(mod), cls)
    return _CACHE[qname]


def channels(qname: str, direction: str) -> dict[str, str]:
    """{채널 id: 채널 이름} — 상속받은 채널까지 포함해서 수집."""
    holder = getattr(widget_class(qname), direction)
    kind = Input if direction == "Inputs" else Output
    found = {}
    for attr in dir(holder):
        value = getattr(holder, attr)
        if isinstance(value, kind):
            found[value.id or value.name] = value.name
    return found


def settings_version(qname: str) -> int | None:
    return getattr(widget_class(qname), "settings_version", None)


def recent_path(relpath: str) -> RecentPath:
    """워크플로우 파일 기준 상대경로로 저장 → 폴더째 옮겨도 열린다."""
    return RecentPath(str((WF_DIR / relpath).resolve()), "basedir", relpath)


def abs_path(relpath: str) -> str:
    return str((WF_DIR / relpath).resolve())


# ─────────────────────────────────────────────────────────────────────────────
# 위젯 단축 이름
# ─────────────────────────────────────────────────────────────────────────────
W = {
    "File":        "Orange.widgets.data.owfile.OWFile",
    "Table":       "Orange.widgets.data.owtable.OWTable",
    "DataInfo":    "Orange.widgets.data.owdatainfo.OWDataInfo",
    "ColStats":    "Orange.widgets.data.owfeaturestatistics.OWFeatureStatistics",
    "Distrib":     "Orange.widgets.visualize.owdistributions.OWDistributions",
    "BoxPlot":     "Orange.widgets.visualize.owboxplot.OWBoxPlot",
    "Violin":      "Orange.widgets.visualize.owviolinplot.OWViolinPlot",
    "Scatter":     "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
    "BarPlot":     "Orange.widgets.visualize.owbarplot.OWBarPlot",
    "Correlations": "Orange.widgets.data.owcorrelations.OWCorrelations",
    "Sieve":       "Orange.widgets.visualize.owsieve.OWSieveDiagram",
    "Mosaic":      "Orange.widgets.visualize.owmosaic.OWMosaicDisplay",
    "HeatMap":     "Orange.widgets.visualize.owheatmap.OWHeatMap",
    "SelectRows":  "Orange.widgets.data.owselectrows.OWSelectRows",
    "SelectCols":  "Orange.widgets.data.owselectcolumns.OWSelectAttributes",
    "Pivot":       "Orange.widgets.data.owpivot.OWPivot",
    "GroupBy":     "Orange.widgets.data.owgroupby.OWGroupBy",
    "Impute":      "Orange.widgets.data.owimpute.OWImpute",
    "Purge":       "Orange.widgets.data.owpurgedomain.OWPurgeDomain",
    "Script":      "Orange.widgets.data.owpythonscript.OWPythonScript",
    "Sampler":     "Orange.widgets.data.owdatasampler.OWDataSampler",
    "Rank":        "Orange.widgets.data.owrank.OWRank",
    "Save":        "Orange.widgets.data.owsave.OWSave",
    "CreateInst":  "Orange.widgets.data.owcreateinstance.OWCreateInstance",
    "TestScore":   "Orange.widgets.evaluate.owtestandscore.OWTestAndScore",
    "Confusion":   "Orange.widgets.evaluate.owconfusionmatrix.OWConfusionMatrix",
    "ROC":         "Orange.widgets.evaluate.owrocanalysis.OWROCAnalysis",
    "Lift":        "Orange.widgets.evaluate.owliftcurve.OWLiftCurve",
    "Calibration": "Orange.widgets.evaluate.owcalibrationplot.OWCalibrationPlot",
    "Predictions": "Orange.widgets.evaluate.owpredictions.OWPredictions",
    "ParamFit":    "Orange.widgets.evaluate.owparameterfitter.OWParameterFitter",
    "Constant":    "Orange.widgets.model.owconstant.OWConstant",
    "Logistic":    "Orange.widgets.model.owlogisticregression.OWLogisticRegression",
    "NaiveBayes":  "Orange.widgets.model.ownaivebayes.OWNaiveBayes",
    "RandomForest": "Orange.widgets.model.owrandomforest.OWRandomForest",
    "GBoost":      "Orange.widgets.model.owgradientboosting.OWGradientBoosting",
    "Tree":        "Orange.widgets.model.owtree.OWTreeLearner",
    "kNN":         "Orange.widgets.model.owknn.OWKNNLearner",
    "Calibrated":  "Orange.widgets.model.owcalibratedlearner.OWCalibratedLearner",
    "SaveModel":   "Orange.widgets.model.owsavemodel.OWSaveModel",
    "LoadModel":   "Orange.widgets.model.owloadmodel.OWLoadModel",
    "Nomogram":    "Orange.widgets.visualize.ownomogram.OWNomogram",
    "TreeViewer":  "Orange.widgets.visualize.owtreeviewer.OWTreeGraph",
    "PythForest":  "Orange.widgets.visualize.owpythagoreanforest.OWPythagoreanForest",
    "LinProj":     "Orange.widgets.visualize.owlinearprojection.OWLinearProjection",
    "FreeViz":     "Orange.widgets.visualize.owfreeviz.OWFreeViz",
    "Radviz":      "Orange.widgets.visualize.owradviz.OWRadviz",
    "tSNE":        "Orange.widgets.unsupervised.owtsne.OWtSNE",
    "PCA":         "Orange.widgets.unsupervised.owpca.OWPCA",
    "kMeans":      "Orange.widgets.unsupervised.owkmeans.OWKMeans",
}


# ─────────────────────────────────────────────────────────────────────────────
# 스킴 빌더
# ─────────────────────────────────────────────────────────────────────────────
class Scheme:
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.nodes: list[dict] = []
        self.links: list[dict] = []
        self.notes: list[dict] = []
        self._keys: dict[str, int] = {}

    def add(self, key: str, kind: str, title: str, pos: tuple[float, float],
            props: dict | None = None) -> str:
        qname = W[kind]
        props = dict(props or {})
        version = settings_version(qname)
        if version is not None:
            props.setdefault("__version__", version)
        self._keys[key] = len(self.nodes)
        self.nodes.append(dict(
            id=str(len(self.nodes)), kind=kind, qname=qname,
            name=widget_class(qname).name, title=title, pos=pos,
            props=props or None,
        ))
        return key

    def link(self, src: str, out_id: str, dst: str, in_id: str) -> None:
        src_q, dst_q = self.nodes[self._keys[src]]["qname"], self.nodes[self._keys[dst]]["qname"]
        outs, ins = channels(src_q, "Outputs"), channels(dst_q, "Inputs")
        assert out_id in outs, f"{src}: 출력 채널 {out_id!r} 없음 · 가능: {sorted(outs)}"
        assert in_id in ins, f"{dst}: 입력 채널 {in_id!r} 없음 · 가능: {sorted(ins)}"
        self.links.append(dict(
            id=str(len(self.links)),
            source_node_id=str(self._keys[src]), sink_node_id=str(self._keys[dst]),
            source_channel=outs[out_id], source_channel_id=out_id,
            sink_channel=ins[in_id], sink_channel_id=in_id,
        ))

    def note(self, x: float, y: float, w: float, h: float, text: str,
             size: int = FONT[1]) -> None:
        self.notes.append(dict(rect=(x, y, w, h), text=text.strip("\n"), size=size))

    # ── 직렬화 ────────────────────────────────────────────────────────────
    def to_xml(self) -> str:
        out = ['<?xml version=\'1.0\' encoding=\'utf-8\'?>',
               f'<scheme version="2.0" title="{escape(self.title)}" '
               f'description="{escape(self.description)}">',
               '\t<nodes>']
        for n in self.nodes:
            out.append(
                f'\t\t<node id="{n["id"]}" name="{escape(n["name"])}" '
                f'qualified_name="{n["qname"]}" project_name="Orange3" version="" '
                f'title="{escape(n["title"])}" position="{n["pos"][0]!r}, {n["pos"][1]!r}" />')
        out += ['\t</nodes>', '\t<links>']
        for l in self.links:
            out.append(
                f'\t\t<link id="{l["id"]}" source_node_id="{l["source_node_id"]}" '
                f'sink_node_id="{l["sink_node_id"]}" '
                f'source_channel="{escape(l["source_channel"])}" '
                f'sink_channel="{escape(l["sink_channel"])}" enabled="true" '
                f'source_channel_id="{l["source_channel_id"]}" '
                f'sink_channel_id="{l["sink_channel_id"]}" />')
        out += ['\t</links>', '\t<annotations>']
        for i, a in enumerate(self.notes):
            out.append(
                f'\t\t<text id="{i}" type="text/plain" rect="{a["rect"]!r}" '
                f'font-family="{FONT[0]}" font-size="{a["size"]}">{escape(a["text"])}</text>')
        out += ['\t</annotations>', '\t<thumbnail />', '\t<node_properties>']
        for n in self.nodes:
            if not n["props"]:
                continue
            data, fmt = dumps(n["props"], format="literal", pickle_fallback=True)
            out.append(
                f'\t\t<properties node_id="{n["id"]}" format="{fmt}">{escape(data)}</properties>')
        out += ['\t</node_properties>', '</scheme>', '']
        return "\n".join(out)

    def save(self, filename: str) -> Path:
        path = WF_DIR / filename
        path.write_text(self.to_xml(), encoding="utf-8")
        ET.parse(path)  # XML 유효성 1차 확인
        print(f"  생성 → {filename}  (위젯 {len(self.nodes)}개 · 연결 {len(self.links)}개)")
        return path


def file_props(relpath: str) -> dict:
    return {"recent_paths": [recent_path(relpath)], "source": 0, "sheet_names": {}}


def script_props(code: str) -> dict:
    code = code.strip("\n") + "\n"
    return {
        "scriptLibrary": [{"name": "실습 코드", "script": code, "filename": None}],
        "currentScriptIndex": 0,
        "scriptText": code,
        "splitterState": None,
        "vimModeEnabled": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 여러 워크플로우에서 재사용하는 파이썬 스크립트
# ─────────────────────────────────────────────────────────────────────────────
SPLIT_SCRIPT = '''
# ── 실습지 문제 3-1 · 그룹 분할 (GroupShuffleSplit) ───────────────────────────
# 장비 1대당 정비이력이 여러 건이므로 무작위로 섞으면 같은 장비가 train/test
# 양쪽에 들어가 모델이 그 장비를 "외워서" 맞힌다(= 그룹 누수).
# Orange 의 Data Sampler 에는 그룹 분할이 없으므로 이 위젯이 그 역할을 한다.
#
# 실행: 위젯을 열고 Run(⌘R / Ctrl+R). 아래 MODE 만 바꾸면 반대쪽이 나온다.
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

MODE = "{mode}"           # "train" 또는 "test"
GROUP_COL = "equipment_id"
TEST_SIZE, SEED = 0.2, 42

data = in_data
groups = np.asarray([str(v) for v in data.get_column(GROUP_COL)])

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(np.arange(len(data)), groups=groups))
idx = train_idx if MODE == "train" else test_idx

out_data = data[idx]
print(f"[{{MODE}}] {{len(out_data)}}행 x {{len(out_data.domain.attributes)}}피처")
print(f"[{{MODE}}] 재고장 비율 = {{np.nanmean(out_data.Y):.3f}}")
print(f"[{{MODE}}] 장비 수 = {{len(set(groups[idx]))}}")
'''

VERIFY_SCRIPT = '''
# ── 실습지 문제 3-1 검증 · 중복 장비 수는 반드시 0 ────────────────────────────
# Data 입력에 Train 스크립트 → Test 스크립트 순서로 두 개를 연결해 두었다.
import numpy as np

train, test = in_datas[0], in_datas[1]
ids = lambda t: {str(v) for v in t.get_column("equipment_id")}

overlap = ids(train) & ids(test)
print("train :", len(train), "행 /", len(ids(train)), "대")
print("test  :", len(test), "행 /", len(ids(test)), "대")
print("중복 장비 수 =", len(overlap), "   <-- 반드시 0")
print("재고장 비율  train %.3f / test %.3f"
      % (np.nanmean(train.Y), np.nanmean(test.Y)))

out_data = train
'''

THRESHOLD_SCRIPT = '''
# ── 실습지 문제 3-4 · 목표 Recall 을 만족하는 임계값 찾기 ─────────────────────
# 예측정비에서 가장 치명적인 오류는 "놓친 재고장(FN)" 이다.
# 기본 임계값 0.5 는 불균형 데이터에서 FN 을 많이 낸다.
#   Data       입력 = Test 데이터
#   Classifier 입력 = 학습된 모델(학습 위젯의 Model 출력)
import numpy as np
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             roc_auc_score, average_precision_score)

TARGET_RECALL = 0.80

data, model = in_data, in_classifier
proba = model(data, model.Probs)[:, 1]
y = data.Y.astype(int)

prec, rec, thr = precision_recall_curve(y, proba)
cand = np.where(rec[:-1] >= TARGET_RECALL)[0]
best_thr = float(thr[cand[np.argmax(prec[cand])]]) if len(cand) else 0.5
print("선택 임계값 =", round(best_thr, 3))

for label, t in [("기본 0.5", 0.5), ("튜닝", best_thr)]:
    tn, fp, fn, tp = confusion_matrix(y, (proba >= t).astype(int)).ravel()
    print(f"[{label:<8}] FN={fn:>3}  FP={fp:>3}  "
          f"Recall={tp / (tp + fn):.3f}  Precision={tp / max(tp + fp, 1):.3f}")

print("ROC-AUC", round(roc_auc_score(y, proba), 3),
      "| PR-AUC", round(average_precision_score(y, proba), 3))

# 생각해보기 — FN 을 줄인 대가로 FP·Precision 이 얼마나 나빠졌는가?
#              군수 현장에서 이 트레이드오프는 받아들일 만한가?
out_data = data
'''

MISSING_SCRIPT = '''
# ── 실습지 문제 3-6 · 예방정비 결측이 "정보"인지 확인 ─────────────────────────
# fault_code 결측은 무작위 결측이 아니라 "예방정비" 를 뜻하는 정보성 결측(MNAR).
import numpy as np

data = in_data
mt = data.domain["maintenance_type"]
fc = data.domain["fault_code"]

is_prev = np.asarray(data.get_column(mt)) == mt.values.index("예방정비")
miss = np.isnan(np.asarray(data.get_column(fc), dtype=float))

print("fault_code 결측 <=> 예방정비 ?", bool(np.all(miss == is_prev)))
y = data.Y
print("결측(예방정비) 재고장률 =", round(float(np.nanmean(y[miss])), 3))
print("비결측          재고장률 =", round(float(np.nanmean(y[~miss])), 3))
print()
print("→ 결측 자체가 신호다. 최빈값으로 채우면(mode_impute) 거짓 정보를 주입하는 것이고,")
print("  예방정비 행을 지우면(drop_rows) 평가 모집단 자체가 바뀐다.")
print("  Impute 위젯의 'As a distinct value' 가 실습지의 na_plus_flag 전략에 해당한다.")

out_data = data
'''

IQR_SCRIPT = '''
# ── 실습지 문제 1-6 추가과제 · IQR(Tukey) 이상치 탐지 ────────────────────────
import numpy as np

COL = "downtime_hours"
x = np.asarray(in_data.get_column(COL), dtype=float)

q1, q3 = np.nanpercentile(x, [25, 75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
mask = (x < low) | (x > high)

print(f"{COL}: Q1={q1:.1f} Q3={q3:.1f} IQR={iqr:.1f}")
print(f"경계 [{low:.1f}, {high:.1f}] · 이상치 {mask.sum()}건 "
      f"({mask.sum() / len(x) * 100:.1f}%)")
print("→ 긴급정비처럼 '의미 있는 극단값'일 수 있으니 무조건 제거하지 말 것.")

out_data = in_data[mask]     # 이상치만 내보낸다 (Data Table 로 확인)
'''


# ─────────────────────────────────────────────────────────────────────────────
# 01 — 실습 1 · EDA & 데이터 품질 파악
# ─────────────────────────────────────────────────────────────────────────────
def wf_01_eda() -> Scheme:
    s = Scheme("01 · EDA & 데이터 품질 파악",
               "실습지 실습 1 (문제 1-1 ~ 1-6) 을 Orange 위젯으로 옮긴 워크플로우")

    s.note(40, 20, 640, 120, """
[실습 1] EDA & 데이터 품질 파악  ·  실습지 문제 1-1 ~ 1-6

File 위젯이 data/equipment_maintenance.tab 을 읽는다. 이 파일은 헤더에 타입과 역할이
적혀 있어 failure_within_30days 가 이미 타깃(class), equipment_id / maintenance_date /
group_fold 가 메타로 잡혀 있다.  각 위젯을 더블클릭해 아래 지시대로 설정한다.""", 14)

    s.add("file", "File", "장비 정비이력 (5,000건)", (120, 420), file_props(DATA_TAB))
    s.note(60, 470, 240, 90, """
문제 1-1 · 데이터 로드
shape 는 5000 x 19 (원본 18열 + group_fold).
Info 란에서 결측 개수를 먼저 확인한다.""", 11)

    s.add("table", "Table", "① 원본 데이터 보기", (350, 90))
    s.note(500, 60, 300, 60, "문제 1-1 · df.head()\n행 클릭 → 다른 위젯으로 Subset 전달도 가능.", 11)

    s.add("info", "DataInfo", "② 행/열·결측 요약", (350, 180))
    s.note(500, 150, 300, 60, "문제 1-1 · df.shape / df.info()\n행 수·피처 수·타깃·메타를 확인.", 11)

    s.add("stats", "ColStats", "③ 컬럼별 분포·결측", (350, 270))
    s.note(500, 240, 320, 75, """
문제 1-2 · df.isnull().sum()
Missing 열을 클릭해 내림차순 정렬 →
fault_code 2008 / fault_category 2008 /
days_since_last_maintenance 200.""", 11)

    s.add("dist_target", "Distrib", "④ 타깃 분포 (불균형)", (350, 360))
    s.note(500, 335, 320, 75, """
문제 1-3 · Variable = failure_within_30days
재고장 비율 19.5% → 불균형 데이터.
생각해보기: 왜 accuracy 를 주지표로 쓰면 안 되나?""", 11)

    s.add("dist_type", "Distrib", "⑤ 정비 유형별 비율", (350, 450))
    s.note(500, 425, 320, 75, """
문제 1-4 ① · Variable = maintenance_type,
Split by = failure_within_30days.
예방정비 40.2% / 사후+긴급 59.8%.
(Orange 에는 파이차트가 없어 막대로 본다)""", 11)

    s.add("box", "BoxPlot", "⑥ 정비유형별 가동중단 시간", (350, 545))
    s.note(500, 520, 320, 70, """
문제 1-4 ③ · Variable = downtime_hours,
Subgroups = maintenance_type.
생각해보기: 어느 유형이 가장 길고 산포가 큰가?""", 11)

    s.add("scatter", "Scatter", "⑦ 노후도 vs 재고장", (350, 640))
    s.note(500, 615, 320, 75, """
문제 1-4 ④ · Axis x = equipment_age_years,
y = downtime_hours, Color = failure_within_30days.
생각해보기: 빨강·파랑이 뚜렷이 갈리는가?""", 11)

    s.add("corr", "Correlations", "⑧ 수치형 상관 상위쌍", (350, 735))
    s.add("scatter_corr", "Scatter", "⑨ 선택한 상관쌍 보기", (570, 735))
    s.note(720, 710, 330, 75, """
문제 1-4 ⑤ · 상관행렬 히트맵 대응.
목록에서 한 쌍을 클릭하면 오른쪽 Scatter Plot 이
그 두 변수로 바뀐다.
생각해보기: 타깃과 |r| 이 가장 큰 값도 얼마나 작은가?""", 11)

    s.add("sieve", "Sieve", "⑩ 정비유형 x 재고장", (350, 830))
    s.note(500, 810, 320, 55, """
문제 1-2 생각해보기 · fault_code 가 결측인 행은
어떤 maintenance_type 인가? 를 눈으로 확인.""", 11)

    s.add("sel_fault", "SelectRows", "예방정비 제외", (350, 925))
    s.add("dist_fault", "Distrib", "⑪ 장비유형별 고장 빈도", (570, 925))
    s.note(720, 900, 330, 75, """
문제 1-4 ② · Select Rows 조건:
maintenance_type  is not  예방정비
→ Distributions 에서 Variable = equipment_type.""", 11)

    s.add("impute", "Impute", "결측치 처리", (350, 1020), IMPUTE_PROPS)
    s.add("purge", "Purge", "도메인 정리", (540, 1020))
    s.add("table_clean", "Table", "⑫ 처리 후 확인", (730, 1020))
    s.note(880, 985, 360, 130, """
문제 1-6 · Impute 의 Default method = "As a distinct value".
· fault_code / fault_category → 'N/A' 범주가 새로 생긴다
· days_since_last_maintenance → 대체값 + _def 지시자 컬럼
  (실습지의 is_first_maintenance 플래그와 같은 역할)
Purge Domain 은 그 뒤처리다. As a distinct value 는 결측이 없는
변수에도 'N/A' 범주와 상수 _def 컬럼을 만들어 두는데, Purge Domain 이
쓰이지 않는 범주와 상수 컬럼을 지워 피처가 16개로 정리된다.
(타깃까지 3개 값이 되어 이진 지표가 깨지는 것도 여기서 막는다)
처리 후 결측 합계 = 0 을 Data Table 에서 확인.""", 11)

    s.add("iqr", "Script", "IQR 이상치 탐지", (350, 1130), script_props(IQR_SCRIPT))
    s.add("table_iqr", "Table", "⑬ 이상치 목록", (570, 1130))
    s.note(720, 1110, 330, 55, "문제 1-6 추가과제 · 위젯을 열고 Run(⌘R/Ctrl+R).\ndowntime_hours 의 Tukey 경계와 이상치 건수 출력.", 11)

    for sink in ("table", "info", "stats", "dist_target", "dist_type", "box",
                 "scatter", "corr", "sieve", "sel_fault", "impute", "iqr",
                 "scatter_corr"):
        s.link("file", "data", sink, "data")
    s.link("corr", "features", "scatter_corr", "features")
    s.link("sel_fault", "matching_data", "dist_fault", "data")
    s.link("impute", "data", "purge", "data")
    s.link("purge", "data", "table_clean", "data")
    s.link("iqr", "data", "table_iqr", "data")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 02 — 실습 2 · 분석 대시보드 (Streamlit 대응)
# ─────────────────────────────────────────────────────────────────────────────
def wf_02_dashboard() -> Scheme:
    s = Scheme("02 · 분석 대시보드",
               "실습지 실습 2 (Streamlit 대시보드) 를 Orange 위젯 조합으로 재현")

    s.note(40, 20, 700, 130, """
[실습 2] 분석 대시보드  ·  실습지 문제 2-1

Streamlit 대시보드의 구성요소를 Orange 위젯으로 1:1 대응시켰다.
  사이드바 필터 → Select Rows      |  KPI 카드      → Data Info · Pivot Table
  막대/박스 차트 → Bar Plot·Box Plot |  월별 추이 라인 → Distributions(시간축)
Select Rows 에서 조건을 바꾸면 아래 모든 위젯이 즉시 갱신된다 = 필터 UI.""", 14)

    s.add("file", "File", "장비 정비이력", (120, 430), file_props(DATA_TAB))
    s.add("filter", "SelectRows", "① 필터 (사이드바 대응)", (330, 430))
    s.note(120, 520, 320, 110, """
문제 2-1 ① · 조건 두 개를 추가한다.
  equipment_type  is one of  (원하는 유형)
  severity        is between 0 and 5
조건을 바꿀 때마다 오른쪽 위젯이 모두 갱신 →
Streamlit 의 st.multiselect / st.slider 와 같은 역할.""", 11)

    s.add("kpi", "DataInfo", "② KPI — 건수", (570, 120))
    s.note(720, 95, 320, 60, "문제 2-1 ② · st.metric('정비 건수') 대응.\n필터 후 행 수가 바로 보인다.", 11)

    s.add("pivot", "Pivot", "③ KPI — 유형별 재고장률", (570, 220))
    s.add("bar", "BarPlot", "④ 재고장률 막대", (800, 220))
    s.note(950, 190, 340, 95, """
문제 2-1 ②③ · Pivot Table 설정
  Rows = equipment_type
  Values = failure_within_30days
  Aggregations = Mean
→ Bar Plot 에서 Values 를 그 평균 컬럼으로 지정하면
   '장비유형별 재고장률' 막대가 된다.""", 11)

    s.add("box", "BoxPlot", "⑤ 유형별 정비비용 분포", (570, 340))
    s.note(720, 315, 320, 60, "문제 2-1 ④ · Variable = maintenance_cost,\nSubgroups = equipment_type (px.box 대응).", 11)

    s.add("dist_month", "Distrib", "⑥ 월별 정비 건수 추이", (570, 440))
    s.note(720, 415, 340, 80, """
문제 2-1 ⑤ · Variable = maintenance_date (시간형).
Bin width 를 월 단위로 늘리면 px.line 의 월별 추이와
같은 그림이 된다. Split by = maintenance_type 도 시도.""", 11)

    s.add("violin", "Violin", "⑦ 가동중단 분포(바이올린)", (570, 540))
    s.note(720, 515, 320, 60, "심화 · Variable = downtime_hours,\nSubgroups = maintenance_type. 박스플롯보다 분포 모양이 잘 보인다.", 11)

    s.add("groupby", "GroupBy", "⑧ 부대별 집계", (570, 640))
    s.add("table_group", "Table", "⑨ 집계 결과", (800, 640))
    s.note(950, 615, 330, 75, """
심화 · Group by = unit_code,
Aggregations 에서 failure_within_30days = Mean,
maintenance_cost = Mean 을 체크.
어느 부대의 재고장률이 가장 높은가?""", 11)

    s.add("table", "Table", "⑩ 필터된 원자료", (570, 740))
    s.note(720, 720, 320, 45, "문제 2-1 · st.dataframe(f.head(20)) 대응.", 11)

    s.link("file", "data", "filter", "data")
    for sink in ("kpi", "pivot", "box", "dist_month", "violin", "groupby", "table"):
        s.link("filter", "matching_data", sink, "data")
    s.link("pivot", "grouped_data", "bar", "data")
    s.link("groupby", "data", "table_group", "data")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 공통 블록 — 결측 처리 → 누수 컬럼 선택 → 그룹 분할
# ─────────────────────────────────────────────────────────────────────────────
IMPUTE_PROPS = {"_default_method_index": 3}     # 3 = As a distinct value

SELECT_COLS_NOTE = """
문제 3-1 · 데이터 누수(leakage) 스위치
Select Columns 를 열고 아래 4개를 Features → Ignored 로 옮기면
실습지의 EXCLUDE_LEAKAGE = True 와 같다.
  fault_code · fault_category · severity · maintenance_success
(모두 '고장이 난 뒤에야 아는 사후 정보')
먼저 포함한 채로 돌리고, 다음에 제외해서 점수 차이를 비교한다."""


def add_pipeline_head(s: Scheme, x0: float, y0: float, *, with_verify: bool = True):
    """File → Impute → Select Columns → Train/Test 그룹 분할 블록을 붙인다."""
    s.add("file", "File", "장비 정비이력", (x0, y0), file_props(DATA_TAB))
    s.add("impute", "Impute", "결측 처리 (1-6)", (x0 + 170, y0), IMPUTE_PROPS)
    s.add("purge", "Purge", "도메인 정리", (x0 + 330, y0))
    s.add("cols", "SelectCols", "누수 컬럼 선택 (3-1)", (x0 + 490, y0))
    s.add("train", "Script", "Train — 그룹 분할", (x0 + 680, y0 - 110),
          script_props(SPLIT_SCRIPT.format(mode="train")))
    s.add("test", "Script", "Test — 그룹 분할", (x0 + 680, y0 + 110),
          script_props(SPLIT_SCRIPT.format(mode="test")))

    s.link("file", "data", "impute", "data")
    s.link("impute", "data", "purge", "data")
    s.link("purge", "data", "cols", "data")
    s.link("cols", "data", "train", "data")
    s.link("cols", "data", "test", "data")

    s.note(x0 - 70, y0 + 60, 330, 105, """
문제 1-6 · Impute + Purge Domain (두 개가 한 세트)
Impute 의 Default method = "As a distinct value"
  → 결측이 곧 '예방정비'라는 신호를 보존한다.
Purge Domain 은 그때 딸려 나오는 빈 범주·상수 컬럼을 정리한다.
이 두 개를 거치면 피처가 정확히 16개가 된다.""", 11)
    s.note(x0 + 400, y0 + 90, 300, 110, SELECT_COLS_NOTE, 11)
    s.note(x0 + 650, y0 - 230, 340, 110, """
문제 3-1 ⭐ 가장 중요 · 그룹 분할
Orange 의 Data Sampler 에는 그룹 분할이 없다.
두 Python Script 가 GroupShuffleSplit(test_size=0.2,
random_state=42) 을 같은 시드로 돌려 Train/Test 를
각각 내보낸다. 위젯을 열고 Run(⌘R/Ctrl+R) 한 번씩.""", 11)

    if with_verify:
        s.add("verify", "Script", "중복 장비 검증 (=0)", (x0 + 890, y0),
              script_props(VERIFY_SCRIPT))
        s.link("train", "data", "verify", "data")
        s.link("test", "data", "verify", "data")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — 실습 3-1 · 데이터 분할과 누수
# ─────────────────────────────────────────────────────────────────────────────
def wf_03_split() -> Scheme:
    s = Scheme("03 · 데이터 분할 (그룹 누수 차단)",
               "실습지 문제 3-1 · GroupShuffleSplit 대 무작위 분할")

    s.note(40, 20, 720, 120, """
[실습 3-1] ⭐ 가장 중요한 단계  ·  실습지 문제 3-1

장비 1대당 정비이력이 여러 건이다. 무작위로 섞으면 같은 장비가 train/test 양쪽에
들어가 모델이 그 장비를 "외워서" 맞힌다(그룹 누수). 아래 두 갈래를 모두 돌려
'중복 장비 수'와 나중 워크플로우의 점수 차이를 직접 비교하라.""", 14)

    add_pipeline_head(s, 120, 400)

    s.add("t_train", "Table", "Train 확인", (960, 200))
    s.add("t_test", "Table", "Test 확인", (960, 520))
    s.add("d_train", "Distrib", "Train 타깃 분포", (1150, 290))
    s.add("d_test", "Distrib", "Test 타깃 분포", (1150, 430))
    s.link("train", "data", "t_train", "data")
    s.link("test", "data", "t_test", "data")
    s.link("train", "data", "d_train", "data")
    s.link("test", "data", "d_test", "data")
    s.note(1080, 100, 330, 90, """
✅ 기대 출력
  X_train 약 4,031행 / X_test 약 969행
  중복 장비 수 = 0
  train·test 재고장 비율 모두 약 0.19""", 11)

    # 함정 비교용 무작위 분할
    s.add("sampler", "Sampler", "❌ 무작위 분할 (함정)", (500, 700),
          {"sampling_type": 0, "sampleSizePercentage": 80, "stratify": True,
           "use_seed": True})
    s.add("t_rand", "Table", "무작위 Train", (720, 700))
    s.add("t_rand2", "Table", "무작위 Test", (720, 810))
    s.link("cols", "data", "sampler", "data")
    s.link("sampler", "data_sample", "t_rand", "data")
    s.link("sampler", "remaining_data", "t_rand2", "data")
    s.note(400, 870, 480, 105, """
문제 3-1 (도전) · Data Sampler = 실습지의 train_test_split.
Fixed proportion 80% + Stratify 로 설정하면 '무작위 분할'이 된다.
이 두 출력을 04 워크플로우의 Test and Score 에 물려 AUC 를 비교해 보라.
무작위 쪽이 더 높게 나오는데 — 그게 왜 '함정'인가?""", 11)

    s.add("miss", "Script", "결측이 정보인지 확인 (3-6)", (500, 990),
          script_props(MISSING_SCRIPT))
    s.link("file", "data", "miss", "data")
    s.note(700, 990, 480, 90, """
문제 3-6 · 예방정비 결측치 전략.
Run 하면 'fault_code 결측 <=> 예방정비 = True' 와
결측/비결측 재고장률(약 0.141 vs 0.231)이 나온다.
→ 결측 자체가 신호이므로 최빈값 대체는 거짓 정보 주입이다.""", 11)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 04 — 실습 3-2 · 다중 모델 비교
# ─────────────────────────────────────────────────────────────────────────────
def wf_04_models() -> Scheme:
    s = Scheme("04 · 다중 모델 비교 (불균형 대응)",
               "실습지 문제 3-2 · 베이스라인 + 불균형 보정 + 그룹 CV")

    s.note(40, 20, 780, 145, """
[실습 3-2] 다중 모델 비교  ·  실습지 문제 3-2

체크포인트 4가지
  ① Constant = DummyClassifier(most_frequent) 베이스라인을 반드시 포함한다.
  ② 불균형 보정 — Logistic Regression / Random Forest 의 'Balance class distribution' 체크.
  ③ 주지표는 accuracy 가 아니라 AUC · Recall · F1. (Orange 에 PR-AUC 는 없어 AUC+Recall 로 본다)
  ④ 검증 방식 — 아래 Test and Score 설정 안내 참고.""", 14)

    add_pipeline_head(s, 120, 430, with_verify=False)

    learners = [
        ("m_const", "Constant", "Dummy (베이스라인)", {}),
        ("m_logit", "Logistic", "Logistic (불균형 보정)", {"class_weight": True, "penalty_type": 1, "C_index": 61}),
        ("m_nb", "NaiveBayes", "Naive Bayes", {}),
        ("m_tree", "Tree", "Decision Tree", {}),
        ("m_rf", "RandomForest", "Random Forest (보정)",
         {"n_estimators": 100, "class_weight": True, "use_random_state": True}),
        ("m_gb", "GBoost", "Gradient Boosting", {}),
        ("m_knn", "kNN", "kNN", {}),
    ]
    for i, (key, kind, title, props) in enumerate(learners):
        s.add(key, kind, title, (960, 130 + i * 95), props)

    s.add("score", "TestScore", "Test and Score", (1230, 430), {"resampling": 5, "n_folds": 2})
    s.link("train", "data", "score", "train_data")
    s.link("test", "data", "score", "test_data")
    for key, *_ in learners:
        s.link(key, "learner", "score", "learner")

    s.note(1130, 620, 420, 150, """
문제 3-2 ④ · Test and Score 설정 — 두 가지를 모두 해 보라.
  (A) Test on test data  ← 기본으로 설정해 두었다.
      위에서 그룹 분할한 Train/Test 로 최종 평가한다.
  (B) Cross validation by feature → group_fold 선택
      = 실습지의 GroupKFold(n_splits=5). group_fold 는 장비 단위로
        미리 부여한 5-폴드 메타 컬럼이라 같은 장비가 두 폴드에
        걸치지 않는다. (B) 를 쓸 땐 Data 입력만 있으면 된다.
Target class 를 '1' 로 두어야 Recall/Precision 이 재고장 기준이 된다.""", 11)

    s.add("cm", "Confusion", "혼동행렬", (1450, 200))
    s.add("roc", "ROC", "ROC 곡선", (1450, 320))
    s.add("lift", "Lift", "Performance Curve", (1450, 440))
    s.add("cal", "Calibration", "확률 보정 곡선", (1450, 560))
    for sink in ("cm", "roc", "lift", "cal"):
        s.link("score", "evaluations_results", sink, "evaluation_results")

    s.note(1620, 170, 400, 205, """
✅ 기대 출력 (Test on test data · Target class = 1)
   Constant(Dummy)  AUC 0.500  CA 0.809  Recall 0.000
   Naive Bayes      AUC 0.672  CA 0.745  Recall 0.470
   Logistic(bal)    AUC 0.634  CA 0.624  Recall 0.568
   Random Forest    AUC 0.669  CA 0.813  Recall 0.059
   Gradient Boost   AUC 0.699  CA 0.812  Recall 0.114
· Dummy 의 CA 가 0.809 로 가장 높은 축에 든다 — 재고장을 하나도
  못 맞히면서도. 이것이 불균형에서 accuracy 를 버려야 하는 이유다.
· 혼동행렬에서 FN(놓친 재고장) 칸을 본다. 임계값 0.5 에서 대부분의
  모델이 재고장의 90% 가까이를 놓친다 → 05 워크플로우에서 해결.
· 생각해보기 ① 모든 모델이 Dummy 보다 AUC 가 높은가?
             ② 불균형 보정을 켠 모델은 Recall 이 왜 크게 오르고
                CA 는 왜 떨어지는가?""", 11)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 05 — 실습 3-3 / 3-4 · 튜닝과 임계값
# ─────────────────────────────────────────────────────────────────────────────
def wf_05_tuning() -> Scheme:
    s = Scheme("05 · 튜닝과 임계값 조정",
               "실습지 문제 3-3(하이퍼파라미터) · 3-4(임계값 튜닝)")

    s.note(40, 20, 780, 130, """
[실습 3-3 · 3-4] 튜닝과 임계값  ·  실습지 문제 3-3, 3-4

Orange 에는 Optuna 같은 자동 탐색기가 없다. 대신
  · Parameter Fitter — 파라미터 하나를 범위로 훑어 과적합 지점을 눈으로 찾는다
  · Calibrated Learner — 확률 보정 + 임계값 최적화를 학습기에 씌운다
  · ROC Analysis — 곡선 위에서 임계값을 직접 옮겨 본다
목표 Recall 0.80 기준 임계값은 오른쪽 아래 Python Script 가 계산한다.""", 14)

    add_pipeline_head(s, 120, 430, with_verify=False)

    s.add("gb", "GBoost", "Gradient Boosting", (960, 300))
    s.add("gb2", "GBoost", "GBM — 직접 튜닝", (960, 420))
    s.add("calib", "Calibrated", "임계값 최적화 래퍼", (1160, 420),
          {"calibration": 0, "threshold": 1})
    s.link("train", "data", "gb", "data")
    s.link("train", "data", "gb2", "data")
    s.link("gb2", "learner", "calib", "base_learner")

    s.note(880, 180, 380, 110, """
문제 3-3 · 하이퍼파라미터.
'GBM — 직접 튜닝' 위젯에서 Method = Gradient Boosting (scikit-learn)
또는 Extreme Gradient Boosting (xgboost) 을 고르고
Number of trees / Learning rate / Max depth 를 바꿔 가며
아래 Test and Score 점수 변화를 기록한다.""", 11)
    s.note(1120, 520, 360, 95, """
문제 3-4 · Calibrated Learner
  Calibration = Sigmoid,  Threshold = Optimize F1
→ 0.5 고정이 아니라 데이터에서 임계값을 고르게 한다.
아래 혼동행렬에서 튜닝 전/후 FN 감소를 비교하라.""", 11)

    s.add("fitter", "ParamFit", "Parameter Fitter (Optuna 대응)", (1160, 250))
    s.link("train", "data", "fitter", "data")
    s.link("gb2", "learner", "fitter", "learner")
    s.note(1400, 190, 350, 90, """
문제 3-3 대응 · Parameter 를 고르고 범위를 주면
학습/검증 점수를 함께 그려 준다.
두 곡선이 벌어지기 시작하는 지점이 과적합 시작점.
Optuna 의 탐색 결과를 눈으로 읽는 버전이라고 보면 된다.""", 11)

    s.add("score", "TestScore", "Test and Score", (1380, 420),
          {"resampling": 5, "n_folds": 2})
    s.link("train", "data", "score", "train_data")
    s.link("test", "data", "score", "test_data")
    s.link("gb", "learner", "score", "learner")
    s.link("calib", "learner", "score", "learner")

    s.add("cm", "Confusion", "혼동행렬 (FN 비교)", (1600, 320))
    s.add("roc", "ROC", "ROC — 임계값 이동", (1600, 440))
    s.add("lift", "Lift", "Performance Curve", (1600, 560))
    for sink in ("cm", "roc", "lift"):
        s.link("score", "evaluations_results", sink, "evaluation_results")
    s.note(1760, 300, 340, 120, """
· 혼동행렬 — 두 모델을 번갈아 고르며 FN 칸을 비교.
· ROC Analysis — Target = 1, 'Show performance line' 을 켜고
  비용비를 움직이면 임계값이 이동한다.
  실습지의 '기본 0.5 vs 튜닝 임계값' 비교를 마우스로 하는 셈.""", 11)

    s.add("thr", "Script", "목표 Recall 0.80 임계값", (1380, 700),
          script_props(THRESHOLD_SCRIPT))
    s.link("test", "data", "thr", "data")
    s.link("gb", "model", "thr", "classifier")
    s.note(1600, 690, 440, 190, """
문제 3-4 ⭐ · 위젯을 열고 Run(⌘R/Ctrl+R).
✅ 기대 출력 (Orange 기본 Gradient Boosting 기준)
   선택 임계값 = 0.14 부근
   [기본 0.5] FN=164  FP= 19  Recall=0.114
   [튜닝    ] FN= 32  FP=458  Recall=0.827
   ROC-AUC 0.699 | PR-AUC 0.366
※ 실습지의 '임계값 0.44 · FN 81→34' 는 Optuna 로 튜닝한 XGBoost 기준이라
   숫자 자체는 다르다. 결론은 같다 — 임계값만 내려도 FN 이 급감하고
   Recall 이 목표 0.80 을 넘는다.
생각해보기 — FN 을 줄인 대가로 FP 가 19 → 458 로 늘었다.
             군수 현장에서 이 트레이드오프는 받아들일 만한가?
주의 — 엄밀히는 임계값을 별도 검증셋에서 골라야 한다. 여기서는 실습 단순화로
       test 를 썼기 때문에 점수가 약간 낙관적이다.""", 11)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 06 — 실습 3-5 · 모델 해석
# ─────────────────────────────────────────────────────────────────────────────
def wf_06_interpretation() -> Scheme:
    s = Scheme("06 · 모델 해석 (SHAP 대응)",
               "실습지 문제 3-5 · 피처 중요도와 단일 예측 근거")

    s.note(40, 20, 800, 155, """
[실습 3-5] 모델 해석  ·  실습지 문제 3-5

기본 Orange 에는 SHAP 위젯이 없다. 실습지의 세 그림을 이렇게 대응시킨다.
  bar (평균 |SHAP|)  → Rank        : 여러 점수로 피처 순위를 매긴다
  summary(beeswarm)  → Nomogram    : 피처가 예측을 어느 방향으로 미는지 한눈에
  waterfall(단일 샘플) → Nomogram + Data : 한 건을 골라 기여도를 읽는다
  트리 구조 확인      → Tree Viewer · Pythagorean Forest

진짜 SHAP 을 쓰려면 Options ▸ Add-ons 에서 'Explain' 애드온을 설치하면
Explain Model / Explain Prediction 위젯이 생긴다 (README 참고).""", 14)

    add_pipeline_head(s, 120, 460, with_verify=False)

    s.add("rank", "Rank", "① 피처 중요도 순위", (960, 300))
    s.add("rank_table", "Table", "순위 표", (1180, 300))
    s.add("rf_scorer", "RandomForest", "Random Forest (점수기)", (960, 180),
          {"n_estimators": 100, "class_weight": True, "use_random_state": True})
    s.link("train", "data", "rank", "data")
    s.link("rf_scorer", "learner", "rank", "scorer")
    s.link("rank", "scores", "rank_table", "data")
    s.note(1180, 170, 380, 110, """
문제 3-5 ① · Rank 는 Information Gain, Gini, ANOVA, χ² 등
여러 기준의 중요도를 한 표에 보여준다.
Random Forest 를 Scorer 로 물리면 트리 기반 중요도 열이 추가된다.
✅ 상위에 severity · maintenance_success 가 오면
   그건 '고장 뒤에야 아는 정보'(누수 의심)라는 신호다.""", 11)

    s.add("logit", "Logistic", "Logistic Regression", (960, 430),
          {"class_weight": True})
    s.add("nomogram", "Nomogram", "② 예측 근거 (Nomogram)", (1180, 430))
    s.link("train", "data", "logit", "data")
    s.link("logit", "model", "nomogram", "classifier")
    s.link("test", "data", "nomogram", "data")
    s.note(1180, 520, 400, 120, """
문제 3-5 ② · Nomogram = SHAP waterfall 대응.
· 막대가 길수록 그 피처가 재고장 확률을 크게 민다(= |SHAP| 큼)
· 오른쪽으로 갈수록 위험을 올리는 방향, 왼쪽이면 낮추는 방향
· 아래 Data(Test) 를 연결해 두었으니 인스턴스를 하나 고르면
  '이 건은 왜 위험으로 예측됐나' 를 그대로 읽을 수 있다.""", 11)

    s.add("tree", "Tree", "Decision Tree", (960, 690))
    s.add("treeview", "TreeViewer", "③ 규칙 보기", (1180, 690))
    s.link("train", "data", "tree", "data")
    s.link("tree", "model", "treeview", "tree")

    s.add("rf_view", "PythForest", "④ 숲 전체 보기", (1180, 810))
    s.link("rf_scorer", "model", "rf_view", "random_forest")
    s.note(1400, 690, 360, 100, """
③④ 트리 계열 해석.
Tree Viewer 는 '어떤 조건에서 재고장 확률이 높아지는가'를
규칙으로 보여주고, Pythagorean Forest 는 숲 전체의
트리 모양을 한 화면에 늘어놓아 다양성을 보여준다.""", 11)

    s.add("box_top", "BoxPlot", "⑤ 상위 피처 확인", (960, 900))
    s.link("train", "data", "box_top", "data")
    s.note(1180, 900, 380, 80, """
⑤ Rank 상위 피처를 Variable 로, failure_within_30days 를
Subgroups 로 놓아 재고장 여부에 따라 분포가 실제로
갈리는지 확인한다. 실습 1 의 상관계수와 비교해 보라
— 상관이 약했던 변수가 상위에 오를 수 있다.""", 11)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 07 — 실습 4 · 모델 저장 + 예측 앱
# ─────────────────────────────────────────────────────────────────────────────
def wf_07_predict() -> Scheme:
    s = Scheme("07 · 모델 저장과 예측 앱",
               "실습지 실습 4 (문제 4-1 ~ 4-3) · 저장/로드 + 실시간 예측 + 결과 기록")

    s.note(40, 20, 820, 150, """
[실습 4] 예측 앱 제작 및 자동화  ·  실습지 문제 4-1 ~ 4-3

두 단계로 나뉜다.
  A. 모델 저장 (위쪽 줄)  — joblib.dump(best_model, 'artifacts/model.pkl') 대응
     학습 → Save Model 로 artifacts/model.pkcls 를 만든다. 먼저 한 번 실행해 두어야
     아래 Load Model 이 파일을 찾는다.
  B. 예측 앱 (아래쪽 줄) — predict_app.py 대응
     Create Instance = Streamlit 입력 폼, Predictions = 예측 결과 화면,
     Save Data = predictions.csv 누적 기록.""", 14)

    # A. 학습 → 저장
    add_pipeline_head(s, 120, 330, with_verify=False)
    s.add("gb", "GBoost", "Gradient Boosting", (960, 220))
    s.add("savemodel", "SaveModel", "① 모델 저장 (4-1)", (1180, 220),
          {"filter": "Pickled model (*.pkcls)",
           "stored_path": abs_path("artifacts"), "stored_name": "model.pkcls",
           "auto_save": False})
    s.link("train", "data", "gb", "data")
    s.link("gb", "model", "savemodel", "model")
    s.note(1180, 120, 400, 85, """
문제 4-1 · Save Model 을 열고 Save As... →
workflow/artifacts/model.pkcls 로 저장한다.
Orange 는 전처리(원-핫·스케일링)까지 모델 안에 함께 저장하므로
실습지의 model.pkl + schema.pkl 을 한 파일이 대신한다.""", 11)

    # B. 로드 → 예측
    s.add("loadmodel", "LoadModel", "② 모델 불러오기", (300, 720))
    s.add("createinst", "CreateInst", "③ 입력 폼 (4-2)", (300, 880))
    s.add("pred", "Predictions", "④ 예측 결과", (560, 800))
    s.add("pred_table", "Table", "⑤ 결과 보기", (780, 800))
    s.add("save_pred", "Save", "⑥ predictions.csv 기록 (4-3)", (780, 920),
          {"filter": "Comma-separated values (*.csv)",
           "stored_path": abs_path("reports"), "stored_name": "predictions.csv",
           "add_type_annotations": False, "auto_save": False})
    s.link("file", "data", "createinst", "data")
    s.link("createinst", "data", "pred", "data")
    s.link("loadmodel", "model", "pred", "predictors")
    s.link("pred", "annotated", "pred_table", "data")
    s.link("pred", "annotated", "save_pred", "data")

    s.note(60, 990, 460, 130, """
문제 4-2 ⭐ · Create Instance = Streamlit 입력 폼.
· 위젯을 열면 피처마다 입력칸이 자동 생성된다 (스키마로 폼 자동 생성 대응)
· 초기값 버튼에서 Median 을 고른다 → 실습지의 '기본값 = 학습 데이터 중앙값'.
  0.0 으로 두면 현실에 없는 장비를 예측하게 된다.
· 값을 바꾸면 오른쪽 Predictions 의 확률이 즉시 갱신된다.
문제 4-2 생각해보기 — 확률을 그대로 쓰지 않고 임계값으로 0/1 을 나누는 이유는?""", 11)

    s.note(560, 630, 420, 130, """
문제 4-2 · Predictions 위젯
· 오른쪽 'Show probabilities' 를 켜면 재고장 확률(%)이 보인다.
· Orange 의 기본 판정은 0.5 다. 실습지처럼 튜닝 임계값(예 0.44)으로
  판정하려면 05 워크플로우의 Calibrated Learner(Threshold = Optimize F1)
  로 학습한 모델을 저장해서 여기에 불러온다.
문제 4-3 · Save Data 로 결과를 reports/predictions.csv 에 저장한다.""", 11)

    # 배치 예측
    s.add("file_new", "File", "신규 정비건 5건", (300, 1180), file_props(NEW_TAB))
    s.add("pred_batch", "Predictions", "⑦ 배치 예측", (560, 1180))
    s.add("table_batch", "Table", "⑧ 배치 결과", (780, 1180))
    s.link("file_new", "data", "pred_batch", "data")
    s.link("loadmodel", "model", "pred_batch", "predictors")
    s.link("pred_batch", "annotated", "table_batch", "data")
    s.note(300, 1290, 500, 95, """
심화 · 한 건씩이 아니라 파일 단위로 예측하는 경로.
data/equipment_new_input.tab 은 타깃을 '?' 로 비운 신규 5건이다.
현장에서는 이 자리에 '오늘 들어온 정비 요청' 파일을 물리면 된다.
결과를 Save Data 로 내보내면 predictions.csv 누적 기록이 된다.""", 11)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 08 — 선택 Lab · 시각화 고도화
# ─────────────────────────────────────────────────────────────────────────────
def wf_08_lab() -> Scheme:
    s = Scheme("08 · 선택 Lab (시각화 고도화)",
               "실습지 선택 Lab A · 다변량 관계와 투영 시각화")

    s.note(40, 20, 800, 130, """
[선택 Lab A] 시각화 고도화  ·  실습지 Lab A-1, A-2

  pairplot(다변량 관계)  → Scatter Plot 의 Find Informative Projections · Linear Projection
  3D 산점도             → FreeViz · Radviz · t-SNE (고차원을 2D 로 눌러 담는다)
  상관 히트맵            → Heat Map
5,000행 전부를 t-SNE 에 넣으면 느리므로 Data Sampler 로 20% 만 뽑아 쓴다.
※ Lab B(MLflow) · Lab C(Prefect) 는 Orange 에 대응 위젯이 없다 — README 참고.""", 14)

    s.add("file", "File", "장비 정비이력", (120, 430), file_props(DATA_TAB))
    s.add("impute", "Impute", "결측 처리", (310, 430), IMPUTE_PROPS)
    s.add("purge", "Purge", "도메인 정리", (470, 430))
    s.add("sampler", "Sampler", "20% 표본 (속도용)", (630, 430),
          {"sampling_type": 0, "sampleSizePercentage": 20, "stratify": True,
           "use_seed": True})
    s.link("file", "data", "impute", "data")
    s.link("impute", "data", "purge", "data")
    s.link("purge", "data", "sampler", "data")

    s.add("scatter", "Scatter", "① 정보량 높은 조합 찾기", (740, 130))
    s.note(900, 100, 380, 85, """
Lab A-1 · pairplot 대응.
'Find Informative Projections' 버튼을 누르면 타깃을 가장 잘
가르는 변수쌍을 점수 순으로 추천해 준다.
seaborn pairplot 을 눈으로 훑는 일을 자동화한 셈.""", 11)

    s.add("heatmap", "HeatMap", "② Heat Map", (740, 240))
    s.note(900, 215, 380, 60, "Lab A-1 · Group by = failure_within_30days 로 두면\n재고장/정상 그룹의 값 패턴 차이가 색으로 드러난다.", 11)

    s.add("linproj", "LinProj", "③ Linear Projection", (740, 350))
    s.add("freeviz", "FreeViz", "④ FreeViz", (740, 460))
    s.add("radviz", "Radviz", "⑤ Radviz", (740, 570))
    s.note(900, 340, 380, 105, """
Lab A-2 · 3D 산점도 대응.
여러 수치형 축을 한 평면에 투영해 보는 방법들이다.
Color = failure_within_30days 로 두고
두 색이 갈리는 방향의 축(변수)이 무엇인지 읽는다.""", 11)

    s.add("tsne", "tSNE", "⑥ t-SNE", (740, 680))
    s.add("pca", "PCA", "⑦ PCA", (740, 790))
    s.add("pca_scatter", "Scatter", "⑧ 주성분 산점도", (960, 790))
    s.note(900, 660, 380, 85, """
⑥ t-SNE — 비선형 축소. 군집이 보이면 '비슷한 정비건'의 무리다.
⑦⑧ PCA — Transformed Data 를 Scatter Plot 에 물려 PC1·PC2 로 본다.
    설명분산 곡선에서 몇 개 성분이면 충분한지 확인.""", 11)

    s.add("kmeans", "kMeans", "⑨ k-Means 군집", (740, 900))
    s.add("kmeans_scatter", "Scatter", "⑩ 군집 색칠", (960, 900))
    s.note(900, 980, 400, 80, """
⑨⑩ 심화 · 군집 결과(Cluster)를 Color 로 두고
재고장 비율이 유난히 높은 군집이 있는지 본다.
있다면 그 군집의 조건이 곧 '위험 장비 프로파일'이다.""", 11)

    s.add("mosaic", "Mosaic", "⑪ Mosaic Display", (740, 1060))
    s.note(900, 1060, 380, 70, """
⑪ 범주형 다변량 · maintenance_type x equipment_type x 타깃 을
한 그림에 놓고 어느 칸의 재고장률이 튀는지 본다.
(plotly 애니메이션 막대의 정적 버전)""", 11)

    for sink in ("scatter", "heatmap", "linproj", "freeviz", "radviz",
                 "tsne", "pca", "kmeans", "mosaic"):
        s.link("sampler", "data_sample", sink, "data")
    s.link("pca", "transformed_data", "pca_scatter", "data")
    s.link("kmeans", "annotated_data", "kmeans_scatter", "data")
    return s


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Orange3 워크플로우 생성")
    builders = [
        ("01_eda.ows", wf_01_eda),
        ("02_dashboard.ows", wf_02_dashboard),
        ("03_split.ows", wf_03_split),
        ("04_model_compare.ows", wf_04_models),
        ("05_tuning_threshold.ows", wf_05_tuning),
        ("06_interpretation.ows", wf_06_interpretation),
        ("07_predict_app.ows", wf_07_predict),
        ("08_lab_visualization.ows", wf_08_lab),
    ]
    for filename, builder in builders:
        builder().save(filename)
    print("완료")


if __name__ == "__main__":
    main()
