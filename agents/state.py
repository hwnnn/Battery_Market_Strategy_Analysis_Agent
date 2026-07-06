"""
BatteryAnalysisState — LangGraph 공유 상태 정의
Distributed Pattern: 라우팅은 edge가 담당, Supervisor 없음
"""

from typing import Any, Dict, List, TypedDict, Annotated
import operator


class BatteryAnalysisState(TypedDict):
    # ── 입력 ──────────────────────────────────────────────────
    query: str                          # 분석 주제 (사용자 입력)

    # ── 문서 관련 ─────────────────────────────────────────────
    vectorstore: Any                    # FAISS 벡터스토어 객체

    # ── 중간 결과 (각 노드가 채워넣음) ────────────────────────
    market_background: str              # T2: 시장 배경 분석 결과
    lges_strategy: str                  # T3: LGES 전략 분석 결과
    catl_strategy: str                  # T4: CATL 전략 분석 결과
    comparison_result: str              # T5: 비교 매트릭스 + SWOT

    # ── 참고 자료 추적 ────────────────────────────────────────
    # 병렬 노드에서 동시에 append → operator.add 로 merge
    references: Annotated[List[Dict], operator.add]

    # ── 보고서 ───────────────────────────────────────────────
    report_draft: str                   # T6: 최종 보고서 (Markdown)

    # ── 제어 ─────────────────────────────────────────────────
    error_messages: Annotated[List[str], operator.add]  # 오류 로그 누적
    rag_iteration: int                  # 호환용 필드. RAG 반복은 rag_tool 내부에서 처리
