from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()


# Declare a chain
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{reference}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | ChatOpenAI(model="gpt-4-turbo-preview")

print("type of chat is ")
print(type(chain))