
from operator import itemgetter
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from app.rag.retrieve_vectordb import retrieve_vectordb as retrive
from langserve.pydantic_v1 import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def get_openai_prompt_principle()->str:
    """
    openai에서 제공한 좋은 프롬프트 원칙
    """
    with open("openai.txt") as f:
        docs = f.read()
    
    return docs

def get_prompt(template_id, message) -> ChatPromptTemplate:
    prompt = None
    
    if "good_prompt" == template_id:
        docs = get_openai_prompt_principle()
        
        template = """Based on the following instrutions, help me write a good prompt TEMPLATE for the following task:

        {task}

        Notably, this prompt TEMPLATE expects that additional information will be provided by the end user of the prompt you are writing. For the piece(s) of information that they are expected to provide, please write the prompt in a format where they can be formatted into as if a Python f-string.

        When you have enough information to create a good prompt, return the prompt in the following format:\n\n```prompt\n\n...\n\n```

        Instructions for a good prompt:

        {instructions}
        """    

        prompt = ChatPromptTemplate.from_messages([("system", template)]).partial(
            instructions=docs
        )
    elif "rag_prompt" == template_id:
        ref_text = retrive(message)
        
        template = """
        아래 [[참조]]내용을 참고해서 [[질문]]에 답변해줘
        
        [[질문]]
        {vector_ref_text}
        
        [[답변]]
        {message}
        
        """
        
        prompt = ChatPromptTemplate.from_messages([("system", template)]).partial(
            vector_ref_text = ref_text
            , message = message
        )
        
    
    return prompt


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

def get_chain(type_id, message):
    print(f"type_id = {type_id} \n message = {message}")
    
    template_id = type_id
    
    prompt = get_prompt(template_id, message)


    _TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
    follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)


    vectorstore = FAISS.from_texts(
        ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }

    
    conversational_qa_chain = (
        _inputs | _context | prompt | ChatOpenAI() | StrOutputParser()
    )
    chain = conversational_qa_chain.with_types(input_type=ChatHistory)
    
    chain.get_graph().print_ascii()
    
    return chain #RunnableParallel(chain=chain)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer