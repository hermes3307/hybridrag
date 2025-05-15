import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import logging

# 최신 huggingface 임베딩 시도
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 디렉토리 설정
BASE_DIR = os.path.join(os.getcwd(), "altibase_rag")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

# 디렉토리 생성
for dir_path in [BASE_DIR, RAW_DIR, PROCESSED_DIR, VECTOR_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class AltibaseScraper:
    """Altibase 케이스 스터디 웹페이지 스크래핑 클래스"""
    
    def __init__(self, base_url="https://altibase.com/kr/learn/case_list.php"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.session = requests.Session()
    
    def fetch_page(self, idx, cate):
        """특정 idx와 cate 값으로 페이지 가져오기"""
        url = f"{self.base_url}?bgu=view&idx={idx}&cate={cate}"
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # HTTP 오류 발생시 예외 발생
            
            # 올바른 페이지인지 확인 (제목이 있는지 체크)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.select_one('.con_tit')
            
            if title and title.text.strip():
                logger.info(f"Successfully fetched page: idx={idx}, cate={cate}")
                return response.text
            else:
                logger.warning(f"Page exists but no content found: idx={idx}, cate={cate}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error fetching page: idx={idx}, cate={cate}, error={str(e)}")
            return None
    
    def extract_content(self, html_content):
        """HTML에서 케이스 스터디 내용 추출"""
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 제목 추출
        title_elem = soup.select_one('.con_tit')
        title = title_elem.text.strip() if title_elem else "제목 없음"
        
        # 날짜 추출
        date_elem = soup.select_one('.txt_date')
        date = date_elem.text.strip() if date_elem else "날짜 없음"
        
        # 본문 컨테이너 추출
        content_container = soup.select_one('.view_con')
        
        if not content_container:
            return {
                "title": title,
                "date": date,
                "content": "내용 없음"
            }
        
        # 이미지 대체 텍스트 추가
        for img in content_container.find_all('img'):
            alt_text = img.get('alt', '이미지')
            img.replace_with(f"[이미지: {alt_text}]")
        
        # 불필요한 스크립트, 스타일 제거
        for tag in content_container.find_all(['script', 'style']):
            tag.decompose()
        
        # 본문 추출 및 정리
        content = content_container.get_text(separator='\n').strip()
        content = re.sub(r'\n{3,}', '\n\n', content)  # 과도한 줄바꿈 정리
        
        return {
            "title": title,
            "date": date,
            "content": content
        }
    
    def scan_and_save(self, idx_range=(235, 324), cate_range=(3, 3), delay=0):
        """지정된 범위의 idx와 cate 값으로 스캔하고 결과 저장"""
        total_found = 0
        
        for cate in range(cate_range[0], cate_range[1] + 1):
            for idx in tqdm(range(idx_range[0], idx_range[1] + 1), desc=f"Scanning cate={cate}"):
                # 이미 저장된 파일이 있는지 확인
                file_path = os.path.join(RAW_DIR, f"altibase_case_{idx}_{cate}.json")
                if os.path.exists(file_path):
                    logger.info(f"File already exists: {file_path}, skipping...")
                    total_found += 1
                    continue
                
                # 페이지 가져오기
                html_content = self.fetch_page(idx, cate)
                
                if html_content:
                    # 내용 추출
                    data = self.extract_content(html_content)
                    
                    if data and len(data["content"]) > 50:  # 내용이 충분히 있는 경우만 저장
                        # 메타데이터 추가
                        data["idx"] = idx
                        data["cate"] = cate
                        data["url"] = f"{self.base_url}?bgu=view&idx={idx}&cate={cate}"
                        
                        # JSON 파일로 저장
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        
                        total_found += 1
                        logger.info(f"Saved content to {file_path}")
                
                # 서버에 부담을 주지 않기 위한 딜레이
                time.sleep(delay)
        
        logger.info(f"Total case studies found and saved: {total_found}")
        return total_found


class RAGProcessor:
    """RAG 시스템을 위한 텍스트 처리 및 벡터화 클래스"""
    

    def __init__(self, embedding_model_name="jhgan/ko-sroberta-multitask"):
        try:
            # 최신 langchain_huggingface 패키지 사용 시도
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            # 기존 방식으로 폴백
            logger.warning("langchain_huggingface package not found, using deprecated import path")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
            
    def process_documents(self):
        """수집된 문서 처리 및 청크로 분할"""
        all_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        
        all_documents = []
        
        for file_name in tqdm(all_files, desc="Processing documents"):
            file_path = os.path.join(RAW_DIR, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 제목과 본문 결합
                full_text = f"제목: {data['title']}\n\n{data['content']}"
                
                # 메타데이터 준비
                metadata = {
                    "title": data["title"],
                    "date": data["date"],
                    "idx": data.get("idx", 0),
                    "cate": data.get("cate", 0),
                    "url": data.get("url", ""),
                    "source": file_name
                }
                
                # 텍스트 분할
                chunks = self.text_splitter.split_text(full_text)
                
                # 각 청크에 메타데이터 추가
                for i, chunk in enumerate(chunks):
                    doc_with_metadata = {
                        "text": chunk,
                        "metadata": {
                            **metadata,
                            "chunk_id": i,
                            "chunk_count": len(chunks)
                        }
                    }
                    all_documents.append(doc_with_metadata)
                    
                # 프로세스된 문서 저장
                processed_file_path = os.path.join(PROCESSED_DIR, file_name.replace('.json', '_processed.json'))
                with open(processed_file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": metadata,
                        "chunks": chunks
                    }, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(all_documents)} chunks from {len(all_files)} documents")
        return all_documents
    
    def create_vector_store(self, documents):
        """문서로부터 벡터 스토어 생성"""
        logger.info("Creating vector store...")
        
        # 문서가 비어있는지 확인
        if not documents:
            logger.error("No documents to vectorize. Please scrape some data first.")
            print("오류: 벡터화할 문서가 없습니다. 먼저 데이터를 스크래핑하세요.")
            return None
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # FAISS 벡터 스토어 생성
        try:
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 벡터 스토어 저장
            vector_store.save_local(VECTOR_DIR)
            logger.info(f"Vector store saved to {VECTOR_DIR}")
            
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            print(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
            return None
        
    def load_vector_store(self):
        """저장된 벡터 스토어 로드"""
        logger.info("Loading vector store...")
        
        if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
            logger.error("Vector store not found. Please create it first.")
            return None
            
        vector_store = FAISS.load_local(
            VECTOR_DIR,
            self.embeddings
        )
        
        logger.info("Vector store loaded successfully")
        return vector_store


class GemmaLLM:
    """로컬 Gemma LLM 래퍼 클래스"""
    
    def __init__(self, model_id="google/gemma-3-8b", quantize=True):
        self.model_id = model_id
        self.quantize = quantize
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.llm = None
        
        self._load_model()
        
    def _load_model(self):
        """Gemma 모델 로드"""
        logger.info(f"Loading Gemma model: {self.model_id}")
        
        try:
            # 장치 설정
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # GPU 메모리 확인 (CUDA 사용 시)
            if device == "cuda":
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    free_mem_gb = free_mem / (1024**3)
                    total_mem_gb = total_mem / (1024**3)
                    logger.info(f"GPU memory: {free_mem_gb:.2f}GB free of {total_mem_gb:.2f}GB total")
                    
                    if free_mem_gb < 5 and not self.quantize:
                        logger.warning("Less than 5GB GPU memory available. Enabling quantization.")
                        print("경고: GPU 메모리가 부족합니다. 자동으로 양자화를 활성화합니다.")
                        self.quantize = True
                except Exception as e:
                    logger.warning(f"Could not check GPU memory: {str(e)}")
            
            # 양자화 설정 (메모리 사용량 감소)
            if self.quantize and device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("4비트 양자화 활성화: GPU 메모리 사용량을 줄입니다.")
            else:
                quantization_config = None
                
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=True,
                use_safetensors=True,
            )
            
            # 텍스트 생성 파이프라인 생성
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
            )
            
            # LangChain LLM 래퍼 생성
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            logger.info("Gemma model loaded successfully")
            
            return self.llm
        
        except Exception as e:
            logger.error(f"Error loading Gemma model: {str(e)}")
            print(f"Gemma 모델 로드 중 오류 발생: {str(e)}")
            return None



class GemmaRAG:
    """Gemma 3 모델을 이용한 RAG 시스템"""
    
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        
        # RAG 프롬프트 템플릿
        self.template = """
        <context>
        {context}
        </context>

        당신은 Altibase 케이스 스터디 전문가입니다. 위의 컨텍스트 정보를 바탕으로 아래 질문에 답변해주세요.
        답변을 할 때는 컨텍스트에 있는 정보만 사용하고, 컨텍스트에 없는 내용은 '제공된 정보만으로는 알 수 없습니다'라고 명시해주세요.
        출처 인용 시 제목과 URL을 포함해 주세요.

        질문: {question}

        답변:
        """
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 구성
        self._initialize_rag_chain()
    
    def _initialize_rag_chain(self):
        """RAG 체인 초기화"""
        # 검색기 설정
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 검색 결과 포맷팅 함수
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # RAG 체인 구성
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def answer_question(self, question):
        """질문에 대한 답변 생성"""
        logger.info(f"Answering question: {question}")
        
        try:
            result = self.rag_chain.invoke(question)
            return result
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def interactive_qa(self):
        """대화형 Q&A 세션"""
        print("\n===== Altibase 케이스 스터디 Q&A 시스템 =====")
        print("질문을 입력하세요. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
        
        while True:
            question = input("\n질문: ")
            
            if question.lower() in ['exit', 'quit', '종료']:
                print("프로그램을 종료합니다.")
                break
                
            if not question.strip():
                continue
                
            print("\n답변 생성 중...")
            answer = self.answer_question(question)
            print(f"\n답변: {answer}")

def main():
    """메인 실행 함수"""
    # 1. 웹페이지 스크래핑
    print("Altibase 케이스 스터디 스크래핑을 시작합니다...")
    scrape_new = input("새로운 데이터를 스크래핑 하시겠습니까? (y/n, 기본: n): ").lower() == 'y'
    
    if scrape_new:
        scraper = AltibaseScraper()
        found_count = scraper.scan_and_save(idx_range=(200, 300), cate_range=(3, 3), delay=0.1)
        
        if found_count == 0:
            print("알림: 스크래핑된 문서가 없습니다. 다른 범위를 시도해 보세요.")
            retry = input("다른 범위로 스크래핑을 시도하시겠습니까? (y/n): ").lower() == 'y'
            if retry:
                start_idx = int(input("시작 idx 값 (기본: 1): ") or "1")
                end_idx = int(input("종료 idx 값 (기본: 1000): ") or "1000")
                start_cate = int(input("시작 cate 값 (기본: 1): ") or "1")
                end_cate = int(input("종료 cate 값 (기본: 7): ") or "7")
                scraper.scan_and_save(idx_range=(start_idx, end_idx), cate_range=(start_cate, end_cate), delay=1.0)
    
    # RAW_DIR에 파일이 있는지 확인
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
    if not raw_files:
        print("오류: 처리할 문서가 없습니다. 스크래핑을 먼저 실행하세요.")
        return
    
    # 2. 문서 처리 및 벡터화
    print(f"\n{len(raw_files)}개의 문서를 처리하고 벡터화합니다...")
    processor = RAGProcessor()
    
    # 기존 벡터 스토어 확인
    vector_store = None
    if os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        create_new = input("기존 벡터 스토어가 있습니다. 새로 생성하시겠습니까? (y/n, 기본: n): ").lower() == 'y'
        
        if create_new:
            documents = processor.process_documents()
            if documents:  # 문서가 있는 경우에만 벡터 스토어 생성
                vector_store = processor.create_vector_store(documents)
            else:
                print("오류: 처리된 문서가 없습니다.")
                return
        else:
            vector_store = processor.load_vector_store()
    else:
        documents = processor.process_documents()
        if documents:  # 문서가 있는 경우에만 벡터 스토어 생성
            vector_store = processor.create_vector_store(documents)
        else:
            print("오류: 처리된 문서가 없습니다.")
            return
    
    # 벡터 스토어 생성 실패 시 종료
    if vector_store is None:
        print("벡터 스토어 생성 또는 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 3. Gemma 모델 로드
    print("\nGemma 3 모델을 로드합니다...")
    try:
        gemma_model_id = input("Gemma 모델 ID를 입력하세요 (기본: google/gemma-3-8b): ") or "google/gemma-3-8b"
        use_quantize = input("모델 양자화를 사용하시겠습니까? (y/n, 기본: y): ").lower() != 'n'
        
        gemma_wrapper = GemmaLLM(model_id=gemma_model_id, quantize=use_quantize)
        
        # 모델 로드 실패 시 종료
        if gemma_wrapper.llm is None:
            print("Gemma 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
            return
        
        # 4. RAG 시스템 구축 및 대화형 Q&A 실행
        print("\nGemma 3 RAG 시스템을 초기화합니다...")
        rag_system = GemmaRAG(llm=gemma_wrapper.llm, vector_store=vector_store)
        rag_system.interactive_qa()
        
    except Exception as e:
        logger.error(f"Error loading Gemma model: {str(e)}")
        print(f"Gemma 모델 로드 중 오류 발생: {str(e)}")
        print("\n모델 로드 관련 문제 해결 방법:")
        print("1. Hugging Face에서 모델 접근 권한이 있는지 확인하세요.")
        print("2. 'huggingface-cli login' 명령으로 로그인이 되어 있는지 확인하세요.")
        print("3. 인터넷 연결을 확인하세요.")
        print("4. 더 작은 모델(예: gemma-2b)로 시도해보세요.")


if __name__ == "__main__":
    main()