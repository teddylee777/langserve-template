from operator import itemgetter
from typing import List, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langserve.pydantic_v1 import BaseModel, Field
from app.database import load_vector_database
from app.prompts import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT
from app.utils import combine_documents, format_chat_history, ensure_dict

class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

def create_chain():
    retriever = load_vector_database()

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough().assign(
            chat_history=lambda x: format_chat_history(ensure_dict(x).get("chat_history", []))
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )

    _context = {
        "context": itemgetter("standalone_question") | retriever | combine_documents,
        "question": lambda x: x["standalone_question"],
    }

    claude_model = ChatAnthropic(model="claude-3-sonnet-20240229")

    conversational_qa_chain = (
        _inputs | _context | ANSWER_PROMPT | claude_model | StrOutputParser()
    )
    
    return conversational_qa_chain.with_types(input_type=ChatHistory)