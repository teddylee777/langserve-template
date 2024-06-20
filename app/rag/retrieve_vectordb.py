import os
import chromadb
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # 경로 수정
from dotenv import load_dotenv
import argparse


def retrieve_vectordb(query):    
    # Chroma 데이터베이스 클라이언트 설정
    client = chromadb.PersistentClient(path="../vectordb/chroma_db")  # 기존 데이터베이스 경로
    
    # 기존 컬렉션 로드
    db = Chroma(
        client=client,
        collection_name="openai_collection",
        embedding_function=OpenAIEmbeddings()
    )

    # 쿼리로 유사한 문서 검색
    docs = db.similarity_search(query)
    
    # 검색 결과 출력
    if docs:
        for doc in docs:
            print(doc.page_content)
            # print("Document ID:", doc.id)
            # print("Document Content:", doc.content)
            # print("Similarity Score:", doc.similarity_score)
            print("-------")
            # print(docs[0].page_content)
    else:
        print("유사한 문서를 찾을 수 없습니다.")
    
    # 결과값 반환
    return docs

