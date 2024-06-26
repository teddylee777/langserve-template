# ============================================================================

# from operator import itemgetter
# from typing import List, Tuple

# from fastapi import FastAPI
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
# from langchain_core.runnables import RunnableMap, RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langserve import add_routes
# from langserve.pydantic_v1 import BaseModel, Field

# from langchain_community.vectorstores import Chroma


# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.chat_models import ChatAnthropic

# from langchain_anthropic import ChatAnthropic


# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from dotenv import load_dotenv
# load_dotenv()


# def log_output(output, message):
#     logger.info(f"{message}: {output}")
#     return output


# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
#     """[대화 이력]과 [질문]이 주어지면 후속 질문을 [단일 질문]으로 바꿔줘.

# [대화 이력]:
# {chat_history}

# [질문]:
# {question}

# [단일 질문]:""")


# ANSWER_PROMPT = ChatPromptTemplate.from_template(
#     """다음 [내용]을 기준으로 [질문]에 답변해줘.
# [내용]: {context}

# [질문]: {question}
# """)



# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


# def _combine_documents(
#     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
# ):
#     """Combine documents into a single string."""
#     doc_strings = [format_document(doc, document_prompt) for doc in docs]
#     return document_separator.join(doc_strings)


# def _format_chat_history(chat_history: List[Tuple]) -> str:
#     """Format chat history into a string."""
#     buffer = ""
#     for dialogue_turn in chat_history:
#         human = "Human: " + dialogue_turn[0]
#         ai = "Assistant: " + dialogue_turn[1]
#         buffer += "\n" + "\n".join([human, ai])
#     return buffer


# # ------------------------------------------------------------------
# # vector database
# # ------------------------------------------------------------------

# # FAISS 인덱스 경로 설정
# faiss_index_path = "/Users/passion1014/project/langchain/langserve-template/vectordb/mycollec"

# # OpenAI 임베딩 초기화
# embeddings = OpenAIEmbeddings()

# # 안전하게 FAISS 인덱스 로드
# try:
#     db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
#     logger.info(f"----- FAISS index loaded from: {faiss_index_path}")
# except ValueError as e:
#     logger.error(f"Error loading FAISS index: {e}")
#     logger.info("Attempting to load index without embeddings...")
#     # 임베딩 없이 로드 시도
#     db = FAISS.load_local(faiss_index_path, allow_dangerous_deserialization=True)
#     logger.info("FAISS index loaded without embeddings. Applying embeddings now.")
#     db.embeddings = embeddings

# # 검색기(Retriever) 설정
# retriever = db.as_retriever(search_kwargs={"k": 1})  # 상위 1개 결과 반환
# logger.info(f"----- FAISS retriever created")

# # 테스트 쿼리 실행
# # test_query = "와타나베가 점심 먹은곳은?"
# # test_results = retriever.get_relevant_documents(test_query)
# # logger.info(f"----- Test retrieval results: {test_results}")

# # FAISS 인덱스의 크기 확인
# index_size = len(db.index_to_docstore_id)
# logger.info(f"----- Number of items in FAISS index: {index_size}")
# # ------------------------------------------------------------------


# def ensure_dict(x):
#     if isinstance(x, dict):
#         return x
#     elif hasattr(x, "__dict__"):
#         return x.__dict__
#     else:
#         return {"input": x}
    
# _inputs = RunnableMap(
#     standalone_question=RunnablePassthrough().assign(
#         # chat_history=lambda x: _format_chat_history(x["chat_history"])
#         chat_history=lambda x: _format_chat_history(ensure_dict(x).get("chat_history", []))
#     )
#     | CONDENSE_QUESTION_PROMPT
#     | ChatOpenAI(temperature=0)
#     | StrOutputParser(),
# )

# _context = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }


# # User input
# class ChatHistory(BaseModel):
#     """Chat history with the bot."""

#     chat_history: List[Tuple[str, str]] = Field(
#         ...,
#         extra={"widget": {"type": "chat", "input": "question"}},
#     )
#     question: str


# # Claude 3 Sonnet 모델 초기화
# claude_model = ChatAnthropic(
#     model="claude-3-sonnet-20240229",
# )

# conversational_qa_chain = (
#     _inputs | _context | ANSWER_PROMPT | claude_model | StrOutputParser()
# )
# chain = conversational_qa_chain.with_types(input_type=ChatHistory)

# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="Spin up a simple api server using Langchain's Runnable interfaces",
# )


# # Adds routes to the app for using the chain under:
# # /invoke
# # /batch
# # /stream
# add_routes(app, chain, path="/prompt", enable_feedback_endpoint=True)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)

# ============================================================================

from fastapi import FastAPI
from langserve import add_routes
from app.config import setup_logging
from app.chain import create_chain

# 로깅 설정
logger = setup_logging()

# FastAPI 앱 생성
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# 체인 생성
chain = create_chain()

# 라우트 추가
add_routes(app, chain, path="/prompt", enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)