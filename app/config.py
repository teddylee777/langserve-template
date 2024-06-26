import logging
from dotenv import load_dotenv

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# 환경 변수 로드
load_dotenv()

# FAISS 인덱스 경로 설정
FAISS_INDEX_PATH = "/Users/passion1014/project/langchain/langserve-template/vectordb/mycollec"