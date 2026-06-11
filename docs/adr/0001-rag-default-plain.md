# ADR 0001 — RAG 기본 경로를 plain으로, Agentic RAG는 토글로 분리

- 상태: 채택 (Accepted)
- 날짜: 2026-06-12
- 맥락 태그: RAG, 검색 품질, 비용/지연

## 결정

RAG 검색의 **기본값을 plain(단발 검색)** 으로 두고, Agentic RAG(grade→rewrite
재검색 루프)는 `RAG_AGENTIC=true` 환경변수로 켜는 **옵트인 기능**으로 분리한다.
구현은 `agents/rag_tool.py`, 측정 근거는 `eval/RESULTS.md`.

## 배경

초기 설계는 "일반 RAG는 검색 품질이 낮아도 그대로 답해 환각 위험"이라는 가정 아래
Agentic RAG(retrieve→grade_documents→rewrite_query 최대 3회)를 기본으로 채택했다.
이 가정을 측정으로 검증했다.

## 측정 (RAGAS·IR, 합성 평가셋, GPT-4o-mini · bge-m3 · FAISS 262청크)

순서대로 다음을 확인했다(상세 수치는 `eval/RESULTS.md`):

1. **n=30**: Agentic가 hard 질의에서 Faithfulness +0.087 등 향상으로 보였다.
2. **n=100으로 확대**: easy/hard의 모든 델타가 **부호 반전** → n=30 결과는 표본 노이즈였음.
   n=100 기준 Agentic는 hard에서 Context Precision −0.033, Recall −0.055로 **오히려 악화**.
3. **원인 진단**: 로그에서 rewrite 쿼리가 주제를 이탈(drift)하는 사례 확인
   (예: "유럽 전기차 판매" → "유럽 인터넷 속도").
4. **개선·재측정**: rewrite 프롬프트에 핵심 엔티티 유지·새 주제 금지·원본 앵커링 적용 →
   Precision +0.027, Recall +0.025 회복(드리프트가 원인이었음 실증). 단 **개선 후에도
   plain을 넘지 못함(잘해야 동률).**

근본 원인: bge-m3 의미 임베딩이 패러프레이즈에 강건해 단일 청크 회수율이 이미 포화
(IR Hit@5 0.87~0.93, 재검색·난이도와 무관). "첫 검색 실패를 복구"한다는 Agentic RAG의
전제 조건이 이 코퍼스에서는 거의 발생하지 않는다.

## 판단

이 코퍼스·검색기 조합에서 grade→rewrite 루프는:

- 품질 **순이득 없음**(잘해야 plain과 동률),
- 매 질의에 grade(+rewrite) LLM 호출 **추가 비용·지연**,
- 부적절한 rewrite 시 **악화 위험**.

따라서 기본값으로 두는 것은 정당화되지 않는다. 그러나 구현 자체는 다음 조건에서 유효하므로
삭제하지 않고 옵트인으로 보존한다:

- 더 크고 노이즈 많은 코퍼스(첫 검색이 recall에서 실제 실패),
- 더 약한/희소(BM25 등) 검색기,
- 멀티홉 등 단발 검색이 부족한 질의 분포.

## 대안과 기각 사유

- **Agentic를 기본 유지**: 측정상 순이득 없음 + 비용/지연 → 기각.
- **Agentic 완전 삭제**: 위 조건에서 재사용 가치 있고, 드리프트 수정으로 무해해짐 → 과함. 기각.
- **plain 기본 + 토글 보존(채택)**: 기본은 단순·저비용, 필요 시 켜는 안전장치.

## 영향

- 기본 실행은 grade/rewrite LLM 호출이 사라져 RAG 단계 비용·지연 감소.
- `RAG_AGENTIC=true`로 언제든 드리프트 수정된 Agentic RAG 사용 가능.
- 평가 재현: `eval/README.md` (eval_ragas는 이 토글을 직접 제어해 plain/agentic 비교).
