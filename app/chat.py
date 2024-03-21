from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# Declare a chain
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신의 이름은 테디 입니다. 사는 곳은 충청도입니다. 충청도 사투리로 답변해 주세요.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | ChatOpenAI(model="gpt-4-turbo-preview")
