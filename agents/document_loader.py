"""
Document Loader Node — LLM 미사용, 순수 데이터 처리
PDF 파싱 → 텍스트 청킹 → BAAI/bge-m3 임베딩 → FAISS 저장
파이프라인 시작 시 1회만 실행
"""

import os
import glob
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents.state import BatteryAnalysisState

# 상수
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
MAX_PAGES = 100
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _load_pdfs(data_dir: str) -> List[Document]:
    """data/ 폴더의 PDF 파일을 PyMuPDF로 파싱, 총 100페이지 제한"""
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))

    if not pdf_files:
        raise FileNotFoundError(
            f"data/ 폴더에 PDF 파일이 없습니다. ({data_dir})\n"
            "분석할 PDF 문서를 data/ 폴더에 넣어주세요."
        )

    documents: List[Document] = []
    total_pages = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            doc = fitz.open(pdf_path)
            file_pages = len(doc)

            # 100페이지 총량 제한
            remaining = MAX_PAGES - total_pages
            if remaining <= 0:
                print(f"[DocumentLoader] 100페이지 한도 도달 — {filename} 스킵")
                doc.close()
                continue

            pages_to_read = min(file_pages, remaining)

            for page_num in range(pages_to_read):
                page = doc[page_num]
                text = page.get_text("text").strip()

                if not text:  # 빈 페이지 스킵
                    continue

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "total_pages": file_pages,
                    }
                ))

            total_pages += pages_to_read
            print(f"[DocumentLoader] {filename}: {pages_to_read}페이지 로드 (누적 {total_pages}p)")
            doc.close()

        except Exception as e:
            print(f"[DocumentLoader] {filename} 파싱 실패: {e}")

    print(f"[DocumentLoader] 총 {len(documents)}개 페이지 로드 완료")
    return documents


def _build_vectorstore(documents: List[Document]) -> FAISS:
    """청킹 → 임베딩 → FAISS 인덱스 생성"""
    # 1. 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[DocumentLoader] 청킹 완료: {len(chunks)}개 청크")

    # 2. 오픈소스 임베딩 (BAAI/bge-m3)
    print(f"[DocumentLoader] 임베딩 모델 로딩: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. FAISS 인덱스 생성
    print("[DocumentLoader] FAISS 인덱스 생성 중...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. 로컬 저장
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"[DocumentLoader] FAISS 저장 완료: {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore_if_exists() -> FAISS | None:
    """이미 구축된 FAISS 인덱스가 있으면 로드하여 반환"""
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None

    print("[DocumentLoader] 기존 FAISS 인덱스 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("[DocumentLoader] FAISS 로드 완료")
    return vectorstore


def document_loader_node(state: BatteryAnalysisState) -> dict:
    """
    LangGraph 노드 함수
    벡터 DB가 이미 존재하면 재사용, 없으면 신규 구축
    """
    # 캐시된 벡터스토어 재사용
    vectorstore = load_vectorstore_if_exists()

    if vectorstore is None:
        documents = _load_pdfs(DATA_DIR)
        vectorstore = _build_vectorstore(documents)

    return {"vectorstore": vectorstore}
