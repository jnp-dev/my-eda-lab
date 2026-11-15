# -*- coding: utf-8 -*-
"""생성된 .ows 를 Orange 의 실제 로더로 열어 검증한다.

  · 모든 위젯 qualified_name 이 레지스트리에 있는가
  · 모든 연결의 채널이 존재하고 타입이 호환되는가
  · 저장된 설정이 해당 위젯의 실제 Setting 이름과 일치하는가
  · File 위젯이 가리키는 파일이 실제로 존재하는가

실행: /Applications/Orange.app/Contents/MacOS/python workflow/tools/validate_workflows.py
"""
from __future__ import annotations

import importlib
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from orangecanvas.registry import WidgetRegistry, CategoryDescription
from orangecanvas.scheme import Scheme as CanvasScheme
from orangecanvas.scheme.readwrite import scheme_load, parse_ows_stream
from orangewidget.settings import Setting
from orangewidget.workflow.discovery import widget_desc_from_module

WF_DIR = Path(__file__).resolve().parents[1]


def build_registry(qnames: set[str]) -> WidgetRegistry:
    reg = WidgetRegistry()
    reg.register_category(CategoryDescription(name="실습"))
    for qname in sorted(qnames):
        module = importlib.import_module(qname.rsplit(".", 1)[0])
        desc = widget_desc_from_module(module)
        desc.category = "실습"
        reg.register_widget(desc)
    return reg


def setting_names(qname: str) -> set[str]:
    mod, cls = qname.rsplit(".", 1)
    widget = getattr(importlib.import_module(mod), cls)
    names = {"__version__", "savedWidgetGeometry", "controlAreaVisible",
             "context_settings"}
    for klass in widget.__mro__:
        names |= {n for n, v in vars(klass).items() if isinstance(v, Setting)}
        # SettingProvider 하위 설정(예: score_table)도 허용
        names |= {n for n, v in vars(klass).items()
                  if type(v).__name__ == "SettingProvider"}
    return names


def validate(path: Path) -> list[str]:
    problems: list[str] = []
    with path.open("rb") as fh:
        desc = parse_ows_stream(fh)
    qnames = {n.qualified_name for n in desc.nodes}
    registry = build_registry(qnames)

    scheme = CanvasScheme()
    errors: list[Exception] = []
    with path.open("rb") as fh:
        scheme_load(scheme, fh, registry=registry, error_handler=errors.append)
    problems += [f"로드 오류: {e!r}" for e in errors]

    for node in scheme.nodes:
        allowed = setting_names(node.description.qualified_name)
        unknown = set(node.properties or {}) - allowed
        if unknown:
            problems.append(f"{node.title}: 알 수 없는 설정 {sorted(unknown)}")
        for recent in (node.properties or {}).get("recent_paths", []):
            target = WF_DIR / recent.relpath
            if not target.exists():
                problems.append(f"{node.title}: 파일 없음 {recent.relpath}")

    for link in scheme.links:                       # 채널 타입 호환성
        try:
            link.set_enabled(True)
        except Exception as exc:                    # pragma: no cover
            problems.append(f"연결 오류: {exc}")

    print(f"{path.name:<28} 위젯 {len(scheme.nodes):>2} · 연결 {len(scheme.links):>2} · "
          f"주석 {len(scheme.annotations):>2} · " + ("OK" if not problems else "문제 발견"))
    return problems


def main() -> int:
    failed = 0
    for path in sorted(WF_DIR.glob("*.ows")):
        for problem in validate(path):
            print("   !", problem)
            failed += 1
    print("\n검증 완료 —", "이상 없음" if not failed else f"{failed}건 확인 필요")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
