from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """[대화 이력]과 [질문]이 주어지면 후속 질문을 [단일 질문]으로 바꿔줘.

[대화 이력]:
{chat_history}

[질문]:
{question}

[단일 질문]:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """다음 [내용]을 기준으로 [질문]에 답변해줘.
[내용]: {context}

[질문]: {question}
""")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")