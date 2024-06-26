from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from app.rag.retrive_vectordb import retrieve_vectordb

# result = retrieve_vectordb("출판사가 어디야?")


system_instruction = """
너는 전문가야 
"""

template = """
아래 내용을 참고해서 답변해줘

{task}

{reference}

{instructions}
"""

prompt = ChatPromptTemplate.from_messages([("system", template)]).partial(
    instructions=system_instruction
)

chain = (
    prompt | ChatOpenAI(model="gpt-4-1106-preview", temperature=0) | StrOutputParser()
)

print("------------------- RAG 조회후 처리 ---------------------")
print(prompt)
