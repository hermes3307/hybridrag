import requests
from bs4 import BeautifulSoup
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Tuple
import re
import json

class LIGNex1RAGInterviewChatbot:
    def __init__(self):
        # 다중 AI 모델 지원
        self.ai_providers = {
            'openai': self.setup_openai(),
            'claude': self.setup_claude(),
            'ollama': self.setup_ollama(),
            'huggingface': self.setup_huggingface()
        }
        
        self.active_provider = self.select_best_provider()
        
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
    
    def advanced_keyword_evaluation(self, question: str, answer: str, context_docs: List[Dict]) -> str:
        """고급 키워드 기반 평가 시스템"""
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # 컨텍스트에서 핵심 키워드 추출
        context_keywords = set()
        context_text = ""
        for doc in context_docs:
            context_text += doc['content'].lower() + " "
            content_words = re.findall(r'\b\w+\b', doc['content'].lower())
            context_keywords.update([word for word in content_words if len(word) > 2])
        
        # 고급 키워드 카테고리
        core_values = {
            'open': ['개방', 'open', '혁신', '창의', '변화', '도전', '새로운', '창조'],
            'positive': ['긍정', 'positive', '열정', '목표', '달성', '성취', '의지', '노력']
        }
        
        culture_values = {
            'pride': ['자부심', 'pride', '자신감', '뿌듯', '성과', '업적'],
            'trust': ['신뢰', 'trust', '믿음', '소통', '협력', '팀워크'],
            'passion': ['열정', 'passion', '몰입', '전념', '집중'],
            'enjoy': ['즐거움', 'enjoy', '재미', '행복', '만족', '균형']
        }
        
        professional_terms = ['전문가', '역량', '성장', '학습', '발전', '향상', '경험', '프로젝트', '성과', '결과']
        company_terms = ['lignex1', 'lig넥스원', '방위산업', '국방', '안보', '군사', '국가']
        
        # 평가 점수 계산
        scores = {
            'context_relevance': 0,
            'core_values': 0,
            'culture_fit': 0,
            'professionalism': 0,
            'company_understanding': 0,
            'specificity': 0,
            'authenticity': 0
        }
        
        feedback_details = []
        
        # 1. 맥락 관련성 평가 (30점)
        context_matches = len([word for word in answer_lower.split() if word in context_keywords])
        if context_matches > 5:
            scores['context_relevance'] = 30
            feedback_details.append("✅ 회사 정보를 매우 잘 활용한 답변")
        elif context_matches > 2:
            scores['context_relevance'] = 20
            feedback_details.append("✅ 회사 정보를 적절히 반영한 답변")
        elif context_matches > 0:
            scores['context_relevance'] = 10
            feedback_details.append("⚡ 회사 정보 활용도를 높여보세요")
        
        # 2. 핵심 가치 부합도 (25점)
        for value_type, keywords in core_values.items():
            if any(keyword in answer_lower for keyword in keywords):
                scores['core_values'] += 12
                feedback_details.append(f"✅ {value_type.upper()} 가치관이 잘 드러남")
        
        # 3. 조직문화 적합성 (20점)
        culture_match_count = 0
        for culture_type, keywords in culture_values.items():
            if any(keyword in answer_lower for keyword in keywords):
                culture_match_count += 1
                scores['culture_fit'] += 5
        
        if culture_match_count >= 2:
            feedback_details.append("✅ 조직문화와 잘 부합하는 답변")
        elif culture_match_count == 1:
            feedback_details.append("⚡ 조직문화 이해도를 더 보여주세요")
        
        # 4. 전문성 (10점)
        prof_matches = sum(1 for term in professional_terms if term in answer_lower)
        if prof_matches >= 3:
            scores['professionalism'] = 10
            feedback_details.append("✅ 전문적인 관점이 돋보임")
        elif prof_matches >= 1:
            scores['professionalism'] = 5
        
        # 5. 회사 이해도 (10점)
        comp_matches = sum(1 for term in company_terms if term in answer_lower)
        if comp_matches >= 1:
            scores['company_understanding'] = 10
            feedback_details.append("✅ 회사에 대한 이해도가 좋음")
        
        # 6. 구체성 평가 (3점)
        experience_indicators = ['경험', '사례', '프로젝트', '활동', '때', '상황', '과정']
        if any(indicator in answer_lower for indicator in experience_indicators):
            scores['specificity'] = 3
            feedback_details.append("✅ 구체적인 경험을 잘 설명함")
        
        # 7. 진정성 평가 (2점)
        if len(answer.strip()) >= 100 and '.' in answer:
            scores['authenticity'] = 2
        elif len(answer.strip()) >= 50:
            scores['authenticity'] = 1
        
        # 부적절한 답변 체크
        inappropriate = ['똥', '바보', '싫다', '모르겠다', '아무거나', '몰라']
        if any(word in answer_lower for word in inappropriate):
            return "❌ 면접에 적절하지 않은 답변입니다. 진솔하고 전문적인 답변을 부탁드립니다."
        
        # 너무 짧은 답변
        if len(answer.strip()) < 20:
            return "📝 답변이 너무 짧습니다. 더 구체적이고 상세한 설명을 부탁드립니다."
        
        # 총점 계산
        total_score = sum(scores.values())
        
        # 등급 결정
        if total_score >= 80:
            grade = "🌟 우수 (A)"
            overall_comment = "매우 인상적인 답변입니다!"
        elif total_score >= 60:
            grade = "👍 양호 (B)"
            overall_comment = "좋은 답변입니다."
        elif total_score >= 40:
            grade = "⚡ 보통 (C)"
            overall_comment = "답변을 더 보완해보세요."
        else:
            grade = "📚 미흡 (D)"
            overall_comment = "회사 정보를 더 활용한 답변이 필요합니다."
        
        # 개선 제안
        improvement_suggestions = []
        if scores['context_relevance'] < 20:
            improvement_suggestions.append("💡 회사 웹사이트 정보를 더 활용해보세요")
        if scores['core_values'] < 15:
            improvement_suggestions.append("💡 OPEN, POSITIVE 가치관을 더 강조해보세요")
        if scores['culture_fit'] < 10:
            improvement_suggestions.append("💡 Pride, Trust, Passion, Enjoy 문화를 언급해보세요")
        if scores['specificity'] < 2:
            improvement_suggestions.append("💡 구체적인 경험이나 사례를 추가해보세요")
        
        # 관련 정보 힌트 제공
        if context_docs:
            relevant_categories = list(set([doc['metadata']['page_type'] for doc in context_docs]))
            if relevant_categories:
                improvement_suggestions.append(f"💡 참고: {', '.join(relevant_categories)} 관련 내용을 더 활용해보세요")
        
        # 결과 포맷팅
        result = f"""📊 고급 RAG 평가 시스템:
{grade} - 총점: {total_score}/100점
{overall_comment}

📈 세부 점수:
• 회사정보 활용: {scores['context_relevance']}/30점
• 핵심가치 부합: {scores['core_values']}/25점  
• 조직문화 적합: {scores['culture_fit']}/20점
• 전문성: {scores['professionalism']}/10점
• 회사 이해도: {scores['company_understanding']}/10점
• 구체성: {scores['specificity']}/3점
• 진정성: {scores['authenticity']}/2점

✅ 강점:
""" + "\n".join(feedback_details)
        
        if improvement_suggestions:
            result += f"\n\n🎯 개선 제안:\n" + "\n".join(improvement_suggestions)
        
        return result

    def setup_openai(self):
        """OpenAI GPT-4 설정"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                import openai
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                print("⚠️ OpenAI 패키지가 없습니다. pip install openai 로 설치하세요.")
                return None
            except Exception as e:
                print(f"OpenAI 설정 실패: {e}")
                return None
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        return None
    
    def setup_claude(self):
        """Claude 설정"""
        api_key = os.getenv('CLAUDE_API_KEY')
        if api_key:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                print(f"Claude 설정 실패: {e}")
                return None
        print("⚠️ CLAUDE_API_KEY가 설정되지 않았습니다.")
        return None
    
    def setup_ollama(self):
        """Ollama (로컬 모델) 설정"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    return True
        except:
            print("⚠️ Ollama 서버가 실행 중이 아닙니다.")
        return None
    
    def setup_huggingface(self):
        """Hugging Face 무료 API 설정"""
        api_key = os.getenv('HUGGINGFACE_API_KEY', 'hf_dummy')
        return api_key
    
    def select_best_provider(self):
        """가장 좋은 AI 제공자 선택"""
        if self.ai_providers['openai']:
            print("✅ OpenAI GPT-4 모델 사용")
            return 'openai'
        elif self.ai_providers['claude']:
            print("✅ Claude 모델 사용")
            return 'claude'
        elif self.ai_providers['ollama']:
            print("✅ Ollama 로컬 모델 사용")
            return 'ollama'
        elif self.ai_providers['huggingface']:
            print("✅ Hugging Face 모델 사용")
            return 'huggingface'
        else:
            print("⚠️ AI 모델 없음. 키워드 기반 모드로 실행")
            return None
    
    def call_ai_model(self, prompt: str, max_tokens: int = 500) -> str:
        """선택된 AI 모델 호출"""
        try:
            if self.active_provider == 'openai':
                response = self.ai_providers['openai'].chat.completions.create(
                    model="gpt-4o",  # Updated to a more current model
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif self.active_provider == 'claude':
                message = self.ai_providers['claude'].messages.create(
                    model="claude-3-opus-20240229",  # Updated to a more generic model
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text if message.content else ""
            
            elif self.active_provider == 'ollama':
                response = requests.post("http://localhost:11434/api/generate", 
                    json={
                        "model": "llama3",  # Simplified model name
                        "prompt": prompt,
                        "stream": False
                    }, timeout=30)
                if response.status_code == 200:
                    return response.json()['response']
                
            elif self.active_provider == 'huggingface':
                headers = {"Authorization": f"Bearer {self.ai_providers['huggingface']}"}
                api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
                
                response = requests.post(api_url, 
                    headers=headers,
                    json={"inputs": prompt[:500]},  # 길이 제한
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
        
        except Exception as e:
            print(f"AI 모델 호출 실패 ({self.active_provider}): {e}")
            self.switch_to_next_provider()
            
        return ""
    
    def switch_to_next_provider(self):
        """다음 AI 제공자로 전환"""
        providers = ['openai', 'claude', 'ollama', 'huggingface']
        if self.active_provider in providers:
            current_idx = providers.index(self.active_provider)
            for i in range(current_idx + 1, len(providers)):
                if self.ai_providers[providers[i]]:
                    self.active_provider = providers[i]
                    print(f"🔄 {providers[i]} 모델로 전환")
                    return
        self.active_provider = None
        
    def extract_text_from_url(self, url: str) -> str:
        """URL에서 텍스트 추출"""
        for attempt in range(3):  # Add retry logic
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
                
            except Exception as e:
                print(f"URL {url} 추출 실패 (시도 {attempt + 1}/3): {e}")
                time.sleep(2)  # Wait before retry
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
        if os.path.exists(self.vector_db_path) and os.path.exists(self.faiss_index_path):
            print("📚 기존 벡터 데이터베이스를 로딩합니다...")
            try:
                self.load_vector_database()
                return
            except Exception as e:
                print(f"벡터 DB 로딩 실패: {e}. 새로 구축합니다...")
        
        print("🔨 벡터 데이터베이스를 구축중입니다...")
        all_chunks = []
        all_metadata = []
        
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
            time.sleep(1)
        
        self.documents = all_chunks
        self.document_metadata = all_metadata
        
        print("🔢 임베딩을 생성중입니다...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        print("🗂️ FAISS 인덱스를 생성중입니다...")
        dimension = embeddings.shape[1]
        self.vector_db = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.vector_db.add(embeddings.astype('float32'))
        
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
        with open(self.vector_db_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.document_metadata
            }, f)
        
        faiss.write_index(self.vector_db, self.faiss_index_path)
    
    def load_vector_database(self):
        """벡터 데이터베이스 로딩"""
        with open(self.vector_db_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_metadata = data['metadata']
        
        self.vector_db = faiss.read_index(self.faiss_index_path)
        print(f"✅ 벡터 DB 로딩 완료: {len(self.documents)}개 문서")
    
    def search_relevant_content(self, query: str, top_k: int = 3) -> List[Dict]:
        """질문과 관련된 내용 검색"""
        if self.vector_db is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
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
        categories = ['인재상', '조직문화', '복리후생', '교육제도', '채용']
        questions = []
        
        for category in categories:
            relevant_docs = self.search_relevant_content(category, top_k=2)
            context = "\n".join([doc['content'] for doc in relevant_docs])
            
            if self.active_provider and context:
                try:
                    prompt = f"""
                    다음 LIG넥스원 {category} 관련 정보를 바탕으로 신입사원 면접 질문 1개를 생성해주세요:
                    
                    {context[:800]}
                    
                    요구사항:
                    - 구체적이고 실용적인 질문
                    - 지원자의 이해도와 적합성을 평가할 수 있는 질문
                    - 한국어로 작성
                    - 질문만 반환 (번호나 추가 설명 없이)
                    
                    예시 형식: "LIG넥스원의 ○○에 대해 설명하고, 본인의 경험과 연결하여 말씀해주세요."
                    """
                    
                    ai_question = self.call_ai_model(prompt, max_tokens=200)
                    
                    if ai_question and len(ai_question.strip()) > 10:
                        clean_question = ai_question.strip().split('\n')[0]
                        clean_question = re.sub(r'^\d+\.\s*', '', clean_question)
                        clean_question = clean_question.strip('"\'')
                        
                        if clean_question:
                            questions.append(clean_question)
                            print(f"✅ {category} 질문 생성 완료")
                        
                except Exception as e:
                    print(f"AI 질문 생성 실패 ({category}): {e}")
        
        premium_base_questions = [
            "LIG넥스원의 인재상인 'OPEN'과 'POSITIVE'에 대해 구체적으로 설명하고, 본인이 이 가치를 실천한 사례를 말씀해주세요.",
            "방위산업이 국가 안보에 미치는 영향에 대한 본인의 견해와 이 분야에서 일하고자 하는 동기를 설명해주세요.",
            "LIG넥스원의 조직문화인 Pride, Trust, Passion, Enjoy 중 본인과 가장 부합하는 가치 2개를 선택하고 그 이유를 구체적으로 말씀해주세요.",
            "혁신과 창조적 사고가 필요했던 프로젝트나 상황에서 본인이 어떤 역할을 했는지 구체적인 사례로 설명해주세요.",
            "LIG넥스원에서 어떤 분야의 전문가가 되고 싶으며, 그 목표를 달성하기 위한 본인의 구체적인 계획은 무엇인가요?",
            "팀 내에서 갈등이나 의견 충돌이 발생했을 때 이를 해결한 경험과 그 과정에서 배운 점을 말씀해주세요.",
            "끝까지 포기하지 않고 목표를 달성한 경험 중 가장 의미 있었던 사례와 그 과정에서의 어려움을 어떻게 극복했는지 설명해주세요."
        ]
        
        while len(questions) < 5:
            remaining_count = 5 - len(questions)
            questions.extend(premium_base_questions[:remaining_count])
            break
        
        return questions[:5]
    
    def evaluate_answer_with_rag(self, question: str, answer: str) -> str:
        """RAG를 사용한 답변 평가"""
        relevant_docs = self.search_relevant_content(question + " " + answer, top_k=3)
        context = "\n".join([f"[{doc['metadata']['page_type']}] {doc['content'][:200]}..." 
                           for doc in relevant_docs])
        
        if self.active_provider and context:
            try:
                prompt = f"""
                LIG넥스원 임원으로서 다음 면접 질문과 답변을 전문적으로 평가해주세요.
                
                관련 회사 정보:
                {context}
                
                면접 질문: {question}
                지원자 답변: {answer}
                
                평가 기준:
                1. 회사 가치관(OPEN, POSITIVE) 및 조직문화와의 부합도
                2. 제공된 회사 정보와의 연관성 및 이해도
                3. 답변의 구체성, 진정성, 논리성
                4. 전문성과 성장 가능성
                
                평가 결과를 다음 형식으로 제공해주세요:
                - 종합 평가: [우수/양호/보통/미흡] 
                - 강점: [구체적인 강점 2-3개]
                - 개선점: [구체적인 개선 제안 1-2개]
                - 추천 점수: [100점 만점 기준]
                
                전문적이고 건설적인 피드백을 제공해주세요.
                """
                
                ai_feedback = self.call_ai_model(prompt, max_tokens=400)
                
                if ai_feedback:
                    return f"🤖 AI 전문 평가 ({self.active_provider.upper()}):\n{ai_feedback}"
                
            except Exception as e:
                print(f"AI 평가 실패: {e}")
        
        return self.advanced_keyword_evaluation(question, answer, relevant_docs)
    
    def keyword_evaluation(self, question: str, answer: str, context_docs: List[Dict]) -> str:
        """향상된 키워드 기반 평가"""
        answer_lower = answer.lower()
        
        context_keywords = set()
        for doc in context_docs:
            content_words = re.findall(r'\b\w+\b', doc['content'].lower())
            context_keywords.update([word for word in content_words if len(word) > 2])
        
        company_keywords = ['lignex1', 'lig넥스원', '방위산업', '국방', '안보']
        value_keywords = ['open', 'positive', '개방', '긍정', 'pride', 'trust', 'passion', 'enjoy']
        professional_keywords = ['전문가', '역량', '성장', '학습', '혁신', '창의', '도전']
        
        score = 0
        feedback_points = []
        
        if len(answer.strip()) < 10:
            return "답변이 너무 짧습니다. 더 구체적이고 상세한 답변을 부탁드립니다."
        
        inappropriate = ['똥', '바보', '싫다', '모르겠다', '아무거나']
        if any(word in answer_lower for word in inappropriate):
            return "❌ 면접에 적절하지 않은 답변입니다. 진솔하고 전문적인 답변을 부탁드립니다."
        
        context_match = len([word for word in answer_lower.split() if word in context_keywords])
        if context_match > 3:
            score += 3
            feedback_points.append("✅ 회사 정보와 잘 연결된 답변입니다")
        elif context_match > 1:
            score += 1
            feedback_points.append("✅ 회사 정보를 일부 반영했습니다")
        
        if any(keyword in answer_lower for keyword in company_keywords):
            score += 2
            feedback_points.append("✅ 회사에 대한 이해도가 좋습니다")
        
        if any(keyword in answer_lower for keyword in value_keywords):
            score += 2
            feedback_points.append("✅ 회사 가치관과 잘 부합합니다")
        
        if any(keyword in answer_lower for keyword in professional_keywords):
            score += 1
            feedback_points.append("✅ 전문적인 사고가 돋보입니다")
        
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
        
        if context_docs:
            hint_info = context_docs[0]['metadata']['page_type']
            feedback_points.append(f"💡 힌트: {hint_info} 관련 내용을 더 활용해보세요")
        
        return f"📊 RAG 기반 키워드 평가:\n{overall}\n" + "\n".join(feedback_points)
    
    def run_interview(self):
        """면접 진행"""
        print("\n" + "="*60)
        print("🎯 LIG넥스원 RAG 면접 시뮬레이션에 오신 것을 환영합니다!")
        print("="*60)
        
        self.build_vector_database()
        
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
    print("🚀 LIG넥스원 고급 RAG 면접 챗봇을 시작합니다!")
    print("🔥 지원되는 AI 모델: OpenAI GPT-4, Claude, Ollama, Hugging Face")
    
    missing_packages = []
    try:
        import sentence_transformers
        import faiss
        import requests
        import bs4
    except ImportError as e:
        missing_packages.append(str(e).split()[-1])
    
    if missing_packages:
        print(f"❌ 필수 패키지가 설치되지 않았습니다: {', '.join(missing_packages)}")
        print("다음 명령어로 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)} sentence-transformers faiss-cpu requests beautifulsoup4")
        return
    
    available_models = []
    if os.getenv('OPENAI_API_KEY'):
        available_models.append("OpenAI GPT-4")
    if os.getenv('CLAUDE_API_KEY'):
        available_models.append("Claude")
    if os.getenv('HUGGINGFACE_API_KEY'):
        available_models.append("Hugging Face")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            available_models.append("Ollama (로컬)")
    except:
        pass
    
    if available_models:
        print(f"\n✅ 사용 가능한 모델: {', '.join(available_models)}")
    else:
        print("\n⚠️ AI 모델이 설정되지 않았습니다. 고급 키워드 평가 모드로 실행됩니다.")
        
    api_choice = input("API 키를 설정하시겠습니까? (o)penai / (c)laude / (h)uggingface / (n)o: ").lower()
    
    if api_choice == 'o':
        key = input("OpenAI API 키 입력: ").strip()
        if key:
            os.environ['OPENAI_API_KEY'] = key
            print("✅ OpenAI 설정 완료")
    elif api_choice == 'c':
        key = input("Claude API 키 입력: ").strip()
        if key:
            os.environ['CLAUDE_API_KEY'] = key
            print("✅ Claude 설정 완료")
    elif api_choice == 'h':
        key = input("Hugging Face API 키 입력 (Enter로 무료 사용): ").strip()
        os.environ['HUGGINGFACE_API_KEY'] = key or 'hf_dummy'
        print("✅ Hugging Face 설정 완료")
    
    try:
        chatbot = LIGNex1RAGInterviewChatbot()
        chatbot.run_interview()
    except KeyboardInterrupt:
        print("\n\n👋 면접을 중단합니다. 수고하셨습니다!")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 문제 해결 방법:")
        print("1. 필요한 패키지 설치: pip install openai sentence-transformers faiss-cpu requests beautifulsoup4")
        print("2. API 키 확인")
        print("3. 인터넷 연결 확인")

if __name__ == "__main__":
    main()