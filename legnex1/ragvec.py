import requests
from bs4 import BeautifulSoup
import os
import anthropic
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Tuple
import re

class LIGNex1RAGInterviewChatbot:
    def __init__(self):
        # Claude API 클라이언트 초기화 (선택사항)
        api_key = os.getenv('CLAUDE_API_KEY')
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.ai_mode = True
            except:
                self.ai_mode = False
        else:
            self.ai_mode = False
        
        # 임베딩 모델 초기화
        print("🔄 임베딩 모델을 로딩중입니다...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 웹사이트 URL 목록
        self.urls = [
            "https://www.lignex1.com/people/talent.do",
            "https://www.lignex1.com/people/life.do", 
            "https://www.lignex1.com/people/welfare.do",
            "https://www.lignex1.com/people/education.do",
            "https://www.lignex1.com/people/jobs.do",
            "https://www.lignex1.com/people/jobsemp.do"
        ]
        
        # 벡터 데이터베이스 관련
        self.vector_db = None
        self.documents = []
        self.document_metadata = []
        
        # 질문 및 상태
        self.questions = []
        self.current_question = 0
        
        # 벡터 DB 파일 경로
        self.vector_db_path = "lignex1_vector_db.pkl"
        self.faiss_index_path = "lignex1_faiss.index"
        
    def extract_text_from_url(self, url: str) -> str:
        """URL에서 텍스트 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 불필요한 태그 제거
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # 텍스트 추출
            text = soup.get_text()
            
            # 정리
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"URL {url} 추출 실패: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """텍스트를 청크 단위로 분할"""
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def build_vector_database(self):
        """벡터 데이터베이스 구축"""
        # 기존 벡터 DB가 있는지 확인
        if os.path.exists(self.vector_db_path) and os.path.exists(self.faiss_index_path):
            print("📚 기존 벡터 데이터베이스를 로딩합니다...")
            self.load_vector_database()
            return
        
        print("🔨 벡터 데이터베이스를 구축중입니다...")
        all_chunks = []
        all_metadata = []
        
        # 각 URL에서 내용 추출 및 청킹
        for url in self.urls:
            print(f"처리중: {url}")
            content = self.extract_text_from_url(url)
            if content:
                chunks = self.chunk_text(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'url': url,
                        'chunk_id': i,
                        'page_type': self.get_page_type(url)
                    })
            time.sleep(1)  # 서버 부하 방지
        
        self.documents = all_chunks
        self.document_metadata = all_metadata
        
        # 임베딩 생성
        print("🔢 임베딩을 생성중입니다...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # FAISS 인덱스 생성
        print("🗂️ FAISS 인덱스를 생성중입니다...")
        dimension = embeddings.shape[1]
        self.vector_db = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도)
        
        # 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)
        self.vector_db.add(embeddings.astype('float32'))
        
        # 벡터 DB 저장
        self.save_vector_database()
        print("✅ 벡터 데이터베이스 구축 완료!")
    
    def get_page_type(self, url: str) -> str:
        """URL에서 페이지 타입 추출"""
        if 'talent' in url:
            return '인재상'
        elif 'life' in url:
            return '조직문화'
        elif 'welfare' in url:
            return '복리후생'
        elif 'education' in url:
            return '교육제도'
        elif 'jobs' in url:
            return '채용정보'
        elif 'jobsemp' in url:
            return '채용전형'
        return '기타'
    
    def save_vector_database(self):
        """벡터 데이터베이스 저장"""
        # 문서와 메타데이터 저장
        with open(self.vector_db_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.vector_db, self.faiss_index_path)
    
    def load_vector_database(self):
        """벡터 데이터베이스 로딩"""
        try:
            # 문서와 메타데이터 로딩
            with open(self.vector_db_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_metadata = data['metadata']
            
            # FAISS 인덱스 로딩
            self.vector_db = faiss.read_index(self.faiss_index_path)
            
            print(f"✅ 벡터 DB 로딩 완료: {len(self.documents)}개 문서")
        except Exception as e:
            print(f"벡터 DB 로딩 실패: {e}")
            self.build_vector_database()
    
    def search_relevant_content(self, query: str, top_k: int = 3) -> List[Dict]:
        """질문과 관련된 내용 검색"""
        if self.vector_db is None:
            return []
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 유사도 검색
        scores, indices = self.vector_db.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.document_metadata[idx],
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def generate_rag_questions(self) -> List[str]:
        """RAG를 사용하여 질문 생성"""
        # 각 카테고리별로 관련 내용 검색
        categories = ['인재상', '조직문화', '복리후생', '교육제도', '채용']
        questions = []
        
        for category in categories:
            relevant_docs = self.search_relevant_content(category, top_k=2)
            context = "\n".join([doc['content'] for doc in relevant_docs])
            
            if self.ai_mode and context:
                # Claude API를 사용한 질문 생성
                try:
                    prompt = f"""
                    다음 LIG넥스원 {category} 관련 정보를 바탕으로 면접 질문 1개를 생성해주세요:
                    
                    {context[:800]}
                    
                    질문은 구체적이고 지원자의 이해도와 적합성을 평가할 수 있어야 합니다.
                    형식: 질문만 반환 (번호 없이)
                    """
                    
                    message = self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    question = message.content[0].text.strip()
                    if question:
                        questions.append(question)
                        
                except Exception as e:
                    print(f"AI 질문 생성 실패 ({category}): {e}")
        
        # 기본 질문 추가
        base_questions = [
            "LIG넥스원의 인재상인 'OPEN'과 'POSITIVE'에 대해 설명하고, 본인이 이에 부합하는 사례를 말씀해주세요.",
            "방위산업 분야에서 일하는 것에 대한 본인의 생각과 각오를 말씀해주세요.",
            "LIG넥스원의 조직문화 중 본인과 가장 부합하는 가치는 무엇이며 그 이유는?",
            "혁신과 창조적 사고를 통해 문제를 해결한 경험이 있다면 구체적으로 말씀해주세요.",
            "LIG넥스원에서 어떤 전문가가 되고 싶으며, 그 이유는 무엇인가요?"
        ]
        
        # AI 생성 질문이 부족하면 기본 질문으로 보완
        if len(questions) < 3:
            questions.extend(base_questions[:5-len(questions)])
        
        return questions[:5]  # 최대 5개 질문
    
    def evaluate_answer_with_rag(self, question: str, answer: str) -> str:
        """RAG를 사용한 답변 평가"""
        # 질문과 관련된 컨텍스트 검색
        relevant_docs = self.search_relevant_content(question + " " + answer, top_k=3)
        context = "\n".join([f"[{doc['metadata']['page_type']}] {doc['content'][:200]}..." 
                           for doc in relevant_docs])
        
        # AI 평가
        if self.ai_mode and context:
            try:
                prompt = f"""
                LIG넥스원 임원으로서 다음 면접 질문과 답변을 평가해주세요.
                
                관련 회사 정보:
                {context}
                
                질문: {question}
                답변: {answer}
                
                평가 기준:
                1. 회사 정보와의 일치도
                2. 답변의 구체성과 진정성
                3. LIG넥스원 가치관 부합도
                
                간단명료한 피드백(3-4문장)을 제공해주세요.
                """
                
                message = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return f"🤖 RAG 기반 AI 평가:\n{message.content[0].text}"
                
            except Exception as e:
                print(f"AI 평가 실패: {e}")
        
        # 키워드 기반 평가
        return self.keyword_evaluation(question, answer, relevant_docs)
    
    def keyword_evaluation(self, question: str, answer: str, context_docs: List[Dict]) -> str:
        """향상된 키워드 기반 평가"""
        answer_lower = answer.lower()
        
        # 컨텍스트에서 핵심 키워드 추출
        context_keywords = set()
        for doc in context_docs:
            content_words = re.findall(r'\b\w+\b', doc['content'].lower())
            context_keywords.update([word for word in content_words if len(word) > 2])
        
        # 기본 키워드들
        company_keywords = ['lignex1', 'lig넥스원', '방위산업', '국방', '안보']
        value_keywords = ['open', 'positive', '개방', '긍정', 'pride', 'trust', 'passion', 'enjoy']
        professional_keywords = ['전문가', '역량', '성장', '학습', '혁신', '창의', '도전']
        
        score = 0
        feedback_points = []
        
        # 답변 길이 체크
        if len(answer.strip()) < 10:
            return "답변이 너무 짧습니다. 더 구체적이고 상세한 답변을 부탁드립니다."
        
        # 부적절한 답변 체크
        inappropriate = ['똥', '바보', '싫다', '모르겠다', '아무거나']
        if any(word in answer_lower for word in inappropriate):
            return "❌ 면접에 적절하지 않은 답변입니다. 진솔하고 전문적인 답변을 부탁드립니다."
        
        # 컨텍스트 관련성 평가
        context_match = len([word for word in answer_lower.split() if word in context_keywords])
        if context_match > 3:
            score += 3
            feedback_points.append("✅ 회사 정보와 잘 연결된 답변입니다")
        elif context_match > 1:
            score += 1
            feedback_points.append("✅ 회사 정보를 일부 반영했습니다")
        
        # 기존 평가 로직들
        if any(keyword in answer_lower for keyword in company_keywords):
            score += 2
            feedback_points.append("✅ 회사에 대한 이해도가 좋습니다")
        
        if any(keyword in answer_lower for keyword in value_keywords):
            score += 2
            feedback_points.append("✅ 회사 가치관과 잘 부합합니다")
        
        if any(keyword in answer_lower for keyword in professional_keywords):
            score += 1
            feedback_points.append("✅ 전문적인 사고가 돋보입니다")
        
        # 종합 평가
        if score >= 6:
            overall = "🌟 매우 우수한 답변입니다!"
        elif score >= 4:
            overall = "👍 좋은 답변입니다."
        elif score >= 2:
            overall = "💡 괜찮은 답변이지만 더 보완해보세요."
        else:
            overall = "📚 회사 정보를 더 반영한 답변을 해보세요."
        
        if not feedback_points:
            feedback_points.append("더 구체적인 사례나 회사 관련 내용을 포함해보세요")
        
        # 관련 정보 힌트 제공
        if context_docs:
            hint_info = context_docs[0]['metadata']['page_type']
            feedback_points.append(f"💡 힌트: {hint_info} 관련 내용을 더 활용해보세요")
        
        return f"📊 RAG 기반 키워드 평가:\n{overall}\n" + "\n".join(feedback_points)
    
    def run_interview(self):
        """면접 진행"""
        print("\n" + "="*60)
        print("🎯 LIG넥스원 RAG 면접 시뮬레이션에 오신 것을 환영합니다!")
        print("="*60)
        
        # 벡터 데이터베이스 구축/로딩
        self.build_vector_database()
        
        # RAG 기반 질문 생성
        print("🧠 RAG 기반 면접 질문을 생성중입니다...")
        self.questions = self.generate_rag_questions()
        
        print(f"\n📋 총 {len(self.questions)}개의 질문이 준비되었습니다.")
        print("각 질문에 성실히 답변해 주세요. (종료하려면 'quit' 입력)")
        print("💡 답변은 LIG넥스원 웹사이트 정보를 바탕으로 평가됩니다.\n")
        
        for i, question in enumerate(self.questions, 1):
            print(f"\n📌 질문 {i}/{len(self.questions)}")
            print("-" * 50)
            print(question)
            print("-" * 50)
            
            answer = input("\n💬 답변: ").strip()
            
            if answer.lower() == 'quit':
                print("\n면접을 종료합니다. 수고하셨습니다!")
                break
                
            if answer:
                print("\n⏳ RAG 시스템으로 답변을 평가중입니다...")
                feedback = self.evaluate_answer_with_rag(question, answer)
                print(f"\n📝 평가 및 피드백:")
                print(feedback)
                print("\n" + "="*50)
            else:
                print("답변을 입력해 주세요.")
                continue
        
        print("\n🎉 RAG 기반 면접 시뮬레이션이 완료되었습니다!")
        print("LIG넥스원 지원에 행운을 빕니다! 💪")

def main():
    print("🚀 LIG넥스원 RAG 면접 챗봇을 시작합니다!")
    
    # 필요 패키지 확인
    try:
        import sentence_transformers
        import faiss
    except ImportError:
        print("❌ 필요한 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 설치해주세요:")
        print("pip install sentence-transformers faiss-cpu")
        return
    
    # API 키 확인 (선택사항)
    api_key = os.getenv('CLAUDE_API_KEY')
    if api_key:
        print("✅ Claude API 키가 감지되었습니다. RAG + AI 평가 모드로 실행됩니다.")
    else:
        print("ℹ️ Claude API 키가 없습니다. RAG + 키워드 평가 모드로 실행됩니다.")
        
        # API 키 입력 옵션
        user_input = input("\nClaude API 키를 입력하시겠습니까? (y/N): ").strip().lower()
        if user_input == 'y':
            user_key = input("API 키를 입력하세요: ").strip()
            if user_key:
                os.environ['CLAUDE_API_KEY'] = user_key
                print("✅ API 키가 설정되었습니다.")
    
    try:
        # RAG 챗봇 실행
        chatbot = LIGNex1RAGInterviewChatbot()
        chatbot.run_interview()
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()