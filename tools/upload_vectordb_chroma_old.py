import os
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # 경로 수정
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import chardet
import argparse


client = chromadb.PersistentClient(path="../vectordb/chroma_db")  # 데이터베이스 경로 지정 # 데이터베이스 클라이언트
collection_name = 'myCollection' # 데이터베이스 컬렉션(또는 테이블)의 이름. 데이터베이스 내에 데이터를 그룹화


# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)


def load_and_detect_encoding(file_path):
    # 1. 파일 로드 및 인코딩 탐지
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        file_encoding = result['encoding']
    
    # 파일 읽기
    with open(file_path, 'r', encoding=file_encoding) as file:
        text = file.read()
    return text


def main():
    # argparse를 사용하여 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='파일 경로를 입력받아 내용을 처리합니다.')
    parser.add_argument('file_path', type=str, help='처리할 파일의 경로')
    parser.add_argument('collection_name', type=str, help='데이터베이스 컬렉션의 이름')
    
    args = parser.parse_args()
    
    # 파일 내용 로드 및 처리
    text = load_and_detect_encoding(args.file_path)
    print("파일 내용을 성공적으로 읽었습니다.")

    # 2. 의미별로 chunk로 나누기
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    chunks = text_splitter.split_text(text)

    # 각 chunk를 문서 객체로 변환
    doc = text_splitter.create_documents(chunks)

    # Chroma에 문서 추가
    db = Chroma.from_documents(documents=doc
                               , embedding=OpenAIEmbeddings()
                               , client=client
                               , collection_name="openai_collection"
                               )

if __name__ == '__main__':
    main()
