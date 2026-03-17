"""
LLM 클라이언트 초기화 — GPT-4o-mini
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """GPT-4o-mini 인스턴스 반환"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=api_key,
    )
