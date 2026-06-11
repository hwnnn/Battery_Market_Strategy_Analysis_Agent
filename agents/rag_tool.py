"""
Agentic RAG Tool
retrieve → grade_documents → [충분] → return
                   └── [부족] → rewrite_query → retrieve (최대 3회)

Tool로 구현: Agent가 직접 호출하는 함수 형태
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

MAX_ITERATIONS = 3
TOP_K = 5


def _retrieve(vectorstore: FAISS, query: str) -> List[Document]:
    """FAISS에서 관련 문서 검색 (top-k=5)"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever.invoke(query)


def _grade_documents(docs: List[Document], query: str, llm: ChatOpenAI) -> bool:
    """
    검색된 문서가 쿼리와 충분히 관련 있는지 LLM으로 평가
    Returns: True(충분) / False(부족)
    """
    if not docs:
        return False

    context = "\n\n".join([d.page_content[:300] for d in docs])
    response = llm.invoke([
        SystemMessage(content=(
            "당신은 검색 결과 관련성 평가자입니다.\n"
            "아래 검색된 문서들이 질문에 답하기에 충분한 정보를 담고 있으면 'yes', "
            "부족하면 'no'만 답하세요."
        )),
        HumanMessage(content=f"질문: {query}\n\n검색된 문서:\n{context}")
    ])

    answer = response.content.strip().lower()
    return "yes" in answer


def _rewrite_query(query: str, llm: ChatOpenAI, attempt: int,
                   original_query: str = None) -> str:
    """관련 문서 부족 시 쿼리를 재구성.

    핵심 대상(고유명사·수치·주제)을 유지하고 새 주제 추가를 금지해
    재작성이 원 질문에서 이탈(drift)하지 않도록 한다. 누적 드리프트를 막기 위해
    원본 질문(original_query)을 앵커로 함께 전달한다.
    """
    anchor = original_query or query
    response = llm.invoke([
        SystemMessage(content=(
            "당신은 검색 쿼리 재작성 전문가입니다. 이전 검색 결과가 부족했습니다.\n"
            "원래 질문의 핵심 대상(고유명사·수치·주제)은 반드시 유지하면서, "
            "표현을 바꾸거나 동일 개념의 동의어를 더해 검색이 잘 되도록 재작성하세요.\n"
            "규칙:\n"
            "- 원래 질문에 없던 새로운 주제·키워드를 추가하지 마세요(주제 이탈 금지).\n"
            "- 한 문장, 핵심 명사 위주로 간결하게 작성하세요.\n"
            "재작성된 쿼리만 출력하세요."
        )),
        HumanMessage(content=(
            f"원래 질문: {anchor}\n"
            f"직전 검색 쿼리 (시도 {attempt}회): {query}"
        ))
    ])
    return response.content.strip()


def _extract_references(docs: List[Document]) -> List[dict]:
    """검색된 문서에서 참고 자료 메타데이터 추출"""
    refs = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            refs.append({
                "type": "pdf",
                "source": source,
                "page": page,
            })
    return refs


def rag_retrieve(
    query: str,
    vectorstore: FAISS,
    llm: ChatOpenAI,
) -> Tuple[str, List[dict]]:
    """
    Agentic RAG 메인 함수
    Returns:
        context (str): 검색된 문서 내용 통합 텍스트
        references (List[dict]): 참고 문서 메타데이터 목록
    """
    current_query = query
    best_docs: List[Document] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        docs = _retrieve(vectorstore, current_query)

        if _grade_documents(docs, query, llm):
            best_docs = docs
            print(f"[RAG] {iteration}회 만에 충분한 문서 확보: {len(docs)}개")
            break
        else:
            best_docs = docs  # 부족하더라도 최선의 결과 보관
            if iteration < MAX_ITERATIONS:
                current_query = _rewrite_query(current_query, llm, iteration,
                                               original_query=query)
                print(f"[RAG] 쿼리 재작성 ({iteration}/{MAX_ITERATIONS}): {current_query[:60]}...")
            else:
                print(f"[RAG] 최대 재시도({MAX_ITERATIONS}회) 도달 — 현재 결과 사용")

    # 컨텍스트 통합
    context_parts = []
    for doc in best_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[출처: {source}, p.{page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "관련 문서를 찾지 못했습니다."
    references = _extract_references(best_docs)

    return context, references
