from typing import List, Tuple
from langchain_core.prompts import format_document
from app.prompts import DEFAULT_DOCUMENT_PROMPT

def combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

def ensure_dict(x):
    if isinstance(x, dict):
        return x
    elif hasattr(x, "__dict__"):
        return x.__dict__
    else:
        return {"input": x}