import requests
from bs4 import BeautifulSoup
import os
import anthropic
import time

class LIGNex1InterviewChatbot:
    def __init__(self):
        # Claude API 클라이언트 초기화
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            print("⚠️ CLAUDE_API_KEY 환경변수가 설정되지 않았습니다.")
            print("다음 방법으로 설정하세요:")
            print("Windows: set CLAUDE_API_KEY=your-api-key")
            print("Linux/Mac: export CLAUDE_API_KEY=your-api-key")
            exit(1)
            
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # 웹사이트 URL 목록
        self.urls = [
            "https://www.lignex1.com/people/talent.do",
            "https://www.lignex1.com/people/life.do", 
            "https://www.lignex1.com/people/welfare.do",
            "https://www.lignex1.com/people/education.do",
            "https://www.lignex1.com/people/jobs.do",
            "https://www.lignex1.com/people/jobsemp.do"
        ]
        
        self.website_content = ""
        self.questions = []
        self.current_question = 0
        
    def extract_text_from_url(self, url):
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
    
    def scrape_all_websites(self):
        """모든 웹사이트에서 내용 추출"""
        print("웹사이트 내용을 추출중입니다...")
        all_content = []
        
        for url in self.urls:
            print(f"추출중: {url}")
            content = self.extract_text_from_url(url)
            if content:
                all_content.append(f"=== {url} ===\n{content}\n")
            time.sleep(1)  # 서버 부하 방지
        
        self.website_content = "\n".join(all_content)
        print("웹사이트 내용 추출 완료!")
    
    def generate_questions(self):
        """웹사이트 내용 기반 면접 질문 생성"""
        # 웹사이트 내용에서 키워드 추출
        content_lower = self.website_content.lower()
        
        # 기본 질문들 (웹사이트 내용 기반)
        base_questions = [
            "1. LIG넥스원의 인재상인 'OPEN'과 'POSITIVE'에 대해 설명하고, 본인이 이에 부합하는 사례를 말씀해주세요.",
            "2. 방위산업 분야에서 일하는 것에 대한 본인의 생각과 각오를 말씀해주세요.",
            "3. LIG넥스원의 조직문화인 Pride, Trust, Passion, Enjoy 중 본인과 가장 부합하는 가치는 무엇이며 그 이유는?",
            "4. 혁신과 창조적 사고를 통해 문제를 해결한 경험이 있다면 구체적으로 말씀해주세요.",
            "5. 끝까지 목표를 달성하기 위해 포기하지 않았던 경험을 말씀해주세요."
        ]
        
        # 추가 질문들 (내용에 따라 동적 생성)
        additional_questions = []
        
        if "교육" in content_lower or "학습" in content_lower:
            additional_questions.append("6. LIG넥스원의 체계적인 교육제도에 대해 어떻게 생각하시며, 어떤 교육을 가장 기대하시나요?")
        
        if "복리후생" in content_lower:
            additional_questions.append("7. 가족 친화적 복리후생 제도에 대한 본인의 생각과 기대는 무엇인가요?")
            
        if "전문가" in content_lower or "specialist" in content_lower:
            additional_questions.append("8. LIG넥스원에서 어떤 분야의 전문가가 되고 싶으며, 그 이유는 무엇인가요?")
        
        # Claude API 사용 시도
        if hasattr(self, 'client') and os.getenv('CLAUDE_API_KEY'):
            try:
                prompt = f"""
                LIG넥스원 임원으로서 다음 웹사이트 내용을 바탕으로 면접 질문 3개를 추가로 생성해주세요:
                
                {self.website_content[:1500]}...
                
                형식: 숫자. 질문내용
                """
                
                message = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                ai_questions = message.content[0].text.strip().split('\n')
                ai_questions = [q.strip() for q in ai_questions if q.strip() and any(char.isdigit() for char in q[:3])]
                additional_questions.extend(ai_questions[:3])
                
            except Exception as e:
                print(f"AI 질문 생성 실패, 기본 질문 사용: {e}")
        
        self.questions = base_questions + additional_questions[:3]  # 최대 8개 질문
    
    def evaluate_answer(self, question, answer):
        """답변을 평가하고 피드백 제공 (AI + 키워드 기반)"""
        
        # 기본 키워드 기반 평가
        def keyword_evaluation(question, answer):
            answer_lower = answer.lower()
            
            # LIG넥스원 관련 키워드
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
            
            # 회사 이해도 평가
            if any(keyword in answer_lower for keyword in company_keywords):
                score += 2
                feedback_points.append("✅ 회사에 대한 이해도가 좋습니다")
            
            # 가치관 일치도 평가
            if any(keyword in answer_lower for keyword in value_keywords):
                score += 2
                feedback_points.append("✅ 회사 가치관과 잘 부합합니다")
            
            # 전문성 평가
            if any(keyword in answer_lower for keyword in professional_keywords):
                score += 1
                feedback_points.append("✅ 전문적인 사고가 돋보입니다")
            
            # 구체적 경험 언급
            experience_words = ['경험', '사례', '프로젝트', '활동', '참여']
            if any(word in answer_lower for word in experience_words):
                score += 1
                feedback_points.append("✅ 구체적인 경험을 잘 설명했습니다")
            
            # 미래 계획/의지
            future_words = ['계획', '목표', '꿈', '희망', '의지', '노력']
            if any(word in answer_lower for word in future_words):
                score += 1
                feedback_points.append("✅ 미래에 대한 명확한 계획이 있습니다")
            
            # 종합 평가
            if score >= 5:
                overall = "🌟 우수한 답변입니다!"
            elif score >= 3:
                overall = "👍 양호한 답변입니다."
            else:
                overall = "💡 답변을 더 구체적으로 보완해보세요."
            
            if not feedback_points:
                feedback_points.append("더 구체적인 사례나 경험을 포함해보세요")
            
            return f"{overall}\n" + "\n".join(feedback_points)
        
        # Claude API 사용 시도
        if hasattr(self, 'client') and os.getenv('CLAUDE_API_KEY'):
            try:
                prompt = f"""
                LIG넥스원 임원으로서 다음 면접 질문과 답변을 평가해 주세요.
                
                질문: {question}
                답변: {answer}
                
                평가 기준:
                1. LIG넥스원의 가치관(OPEN, POSITIVE)과 부합도
                2. 답변의 구체성과 진정성
                3. 회사/업계에 대한 이해도
                
                간단명료한 피드백(3-4문장)을 제공해주세요.
                """
                
                message = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return "🤖 AI 평가:\n" + message.content[0].text
                
            except Exception as e:
                print(f"AI 평가 실패, 기본 평가 사용: {e}")
        
        # 키워드 기반 평가 사용
        return "📊 키워드 기반 평가:\n" + keyword_evaluation(question, answer)
    
    def run_interview(self):
        """면접 진행"""
        print("\n" + "="*60)
        print("🎯 LIG넥스원 면접 시뮬레이션에 오신 것을 환영합니다!")
        print("="*60)
        
        # 웹사이트 내용 추출
        if not self.website_content:
            self.scrape_all_websites()
        
        # 질문 생성
        if not self.questions:
            print("면접 질문을 생성중입니다...")
            self.generate_questions()
        
        print(f"\n📋 총 {len(self.questions)}개의 질문이 준비되었습니다.")
        print("각 질문에 성실히 답변해 주세요. (종료하려면 'quit' 입력)\n")
        
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
                print("\n⏳ 답변을 평가중입니다...")
                feedback = self.evaluate_answer(question, answer)
                print(f"\n📝 평가 및 피드백:")
                print(feedback)
                print("\n" + "="*50)
            else:
                print("답변을 입력해 주세요.")
                continue
        
        print("\n🎉 면접 시뮬레이션이 완료되었습니다!")
        print("LIG넥스원 지원에 행운을 빕니다! 💪")

def main():
    # API 키 확인 및 설정 도움말
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        print("⚠️ CLAUDE_API_KEY 환경변수를 설정해주세요.")
        print("\n설정 방법:")
        print("1. Claude API 콘솔에서 API 키 발급: https://console.anthropic.com/")
        print("2. 환경변수 설정:")
        print("   Windows: set CLAUDE_API_KEY=sk-ant-api03-...")
        print("   Linux/Mac: export CLAUDE_API_KEY=sk-ant-api03-...")
        print("3. 또는 Python에서 직접 설정:")
        print("   os.environ['CLAUDE_API_KEY'] = 'sk-ant-api03-...'")
        
        # 직접 입력 옵션
        user_key = input("\n또는 여기에 API 키를 직접 입력하세요 (Enter로 건너뛰기): ").strip()
        if user_key:
            os.environ['CLAUDE_API_KEY'] = user_key
            print("✅ API 키가 설정되었습니다.")
        else:
            return
    
    try:
        # 챗봇 실행
        chatbot = LIGNex1InterviewChatbot()
        chatbot.run_interview()
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        if "authentication_error" in str(e):
            print("💡 API 키가 올바르지 않습니다. 다시 확인해주세요.")

if __name__ == "__main__":
    main()