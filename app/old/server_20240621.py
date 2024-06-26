from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes

from app.old.chain_20240626 import get_chain
from app.chat import chain as chat_chain



from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# def log_add_routes(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         logging.info(f"add_routes called with args: {args}")
#         logging.info(f"add_routes called with kwargs: {kwargs}")
#         return func(*args, **kwargs)
#     return wrapper


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt/playground")

add_routes(app, get_chain('rag_prompt', 'InputChat'), path="/prompt", enable_feedback_endpoint=True)

add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
