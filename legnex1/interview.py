import streamlit as st
import json
import datetime
import re
from typing import Dict, List, Tuple
import pandas as pd

class LIGNEXInterviewSystem:
    def __init__(self):
        self.questions = [
            {
                "text": "LIG넥스원의 핵심가치 중 '개방(OPEN)'에 대해 어떻게 이해하고 계시며, 본인의 경험 중 개방적 사고로 문제를 해결한 사례가 있다면 말씀해 주세요.",
                "category": "핵심가치 이해도",
                "follow_up": [
                    "그 상황에서 다른 사람들의 의견을 어떻게 수용하셨나요?",
                    "개방적 사고가 어려웠던 순간은 언제였나요?"
                ]
            },
            {
                "text": "방위산업은 국가 안보와 직결되는 중요한 분야입니다. 이러한 책임감이 개인의 업무 수행에 어떤 영향을 미칠 것이라고 생각하시나요?",
                "category": "책임감과 사명감",
                "follow_up": [
                    "압박감과 책임감을 어떻게 관리하실 계획인가요?",
                    "국가를 위한 일이라는 자부심을 어떻게 업무 동기로 연결하시겠나요?"
                ]
            },
            {
                "text": "LIG넥스원은 전체 직원의 50%가 R&D 인력으로 구성된 기술 중심 조직입니다. 지속적인 학습과 기술 발전이 필요한 환경에서 본인만의 성장 전략은 무엇인가요?",
                "category": "학습 지향성",
                "follow_up": [
                    "새로운 기술 트렌드를 어떻게 따라가고 계신가요?",
                    "동료들과 지식을 공유하는 본인만의 방법이 있나요?"
                ]
            },
            {
                "text": "LIG넥스원의 또 다른 핵심가치인 '긍정(POSITIVE)'에 대해 본인의 해석을 들려주시고, 어려운 상황에서도 긍정적 태도를 유지했던 경험을 공유해 주세요.",
                "category": "긍정적 사고",
                "follow_up": [
                    "실패를 성공의 밑거름으로 활용한 경험이 있나요?",
                    "팀원들의 사기가 낮을 때 어떻게 동기부여 하시겠나요?"
                ]
            },
            {
                "text": "LIG넥스원에서 10년 후 본인의 모습을 그려보시고, 회사와 함께 성장하기 위한 본인의 비전을 말씀해 주세요.",
                "category": "비전과 성장 의지",
                "follow_up": [
                    "회사의 발전에 어떤 기여를 하고 싶으신가요?",
                    "개인 성장과 조직 발전의 균형을 어떻게 맞추시겠나요?"
                ]
            }
        ]
        
        self.keywords_by_category = {
            "핵심가치 이해도": ["개방", "협업", "소통", "혁신", "변화", "다양성", "창의", "아이디어", "의견"],
            "책임감과 사명감": ["책임", "국가", "안보", "사명", "의무", "보람", "자부심", "국방", "보안"],
            "학습 지향성": ["학습", "성장", "발전", "기술", "연구", "도전", "공부", "습득", "향상"],
            "긍정적 사고": ["긍정", "열정", "극복", "도전", "성취", "목표", "희망", "자신감", "포기하지"],
            "비전과 성장 의지": ["비전", "목표", "계획", "기여", "발전", "미래", "성장", "꿈", "장기적"]
        }

def initialize_session_state():
    """세션 상태 초기화"""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'follow_up_answers' not in st.session_state:
        st.session_state.follow_up_answers = {}
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'interview_completed' not in st.session_state:
        st.session_state.interview_completed = False
    if 'show_follow_up' not in st.session_state:
        st.session_state.show_follow_up = False
    if 'follow_up_question' not in st.session_state:
        st.session_state.follow_up_question = ""

def evaluate_answer(answer: str, category: str, keywords_by_category: Dict) -> int:
    """답변 평가 함수"""
    if not answer or len(answer.strip()) < 10:
        return 20
    
    score = 0
    answer_lower = answer.lower()
    
    # 기본 점수 (답변 길이 기반)
    score += min(len(answer) / 15, 25)
    
    # 카테고리별 키워드 평가
    keywords = keywords_by_category.get(category, [])
    keyword_score = 0
    
    for keyword in keywords:
        if keyword in answer_lower:
            keyword_score += 8
    
    score += min(keyword_score, 35)
    
    # 구체성 평가
    specificity_indicators = ["예를 들어", "경험", "사례", "때", "상황", "결과", "성과"]
    specificity_score = 0
    
    for indicator in specificity_indicators:
        if indicator in answer:
            specificity_score += 5
    
    score += min(specificity_score, 20)
    
    # 회사 관련 언급
    company_keywords = ["LIG", "넥스원", "방위산업", "국방", "방산"]
    for keyword in company_keywords:
        if keyword in answer:
            score += 5
    
    # 감정적 표현 평가
    emotional_keywords = ["열정", "자신감", "도전", "목표", "꿈", "비전", "희망"]
    for keyword in emotional_keywords:
        if keyword in answer:
            score += 3
    
    return min(round(score), 100)

def get_grade(score: int) -> str:
    """점수에 따른 등급 반환"""
    if score >= 90:
        return "최우수 (A+)"
    elif score >= 85:
        return "우수 (A)"
    elif score >= 80:
        return "양호 (B+)"
    elif score >= 75:
        return "보통 (B)"
    elif score >= 70:
        return "미흡 (C+)"
    elif score >= 60:
        return "부족 (C)"
    else:
        return "개선 필요 (D)"

def generate_feedback(category: str, score: int) -> Tuple[str, str]:
    """카테고리별 피드백 생성"""
    feedback_map = {
        "핵심가치 이해도": {
            "title": "🎯 핵심가치 적합도",
            "high": "LIG넥스원의 핵심가치인 OPEN과 POSITIVE에 대한 깊은 이해를 보여주셨습니다. 실제 경험과 연결한 설명이 매우 인상적이었습니다.",
            "low": "핵심가치에 대한 기본적인 이해는 있으나, 더 구체적인 경험 사례와 연결하여 설명하시면 좋겠습니다."
        },
        "책임감과 사명감": {
            "title": "🛡️ 사명감과 책임감",
            "high": "방위산업의 중요성을 정확히 인식하고 있으며, 국가 안보에 기여하려는 강한 사명감을 확인할 수 있었습니다.",
            "low": "방위산업에 대한 기본적인 이해는 있으나, 개인적인 사명감과 책임감을 더 구체적으로 표현하시면 좋겠습니다."
        },
        "학습 지향성": {
            "title": "📚 성장 의지와 학습 태도",
            "high": "지속적인 학습과 기술 발전에 대한 의지가 명확합니다. R&D 중심 조직에서 필요한 자세를 잘 갖추고 계십니다.",
            "low": "학습에 대한 의지는 있으나, 구체적인 학습 계획이나 방법론을 더 상세히 제시하시면 좋겠습니다."
        },
        "긍정적 사고": {
            "title": "⚡ 긍정적 사고와 도전 정신",
            "high": "어려운 상황에서도 긍정적 태도를 유지하는 능력이 뛰어나며, 실패를 성장의 기회로 인식하는 성숙한 사고를 보여주셨습니다.",
            "low": "긍정적 사고에 대한 이해는 있으나, 구체적인 경험 사례를 통해 더 설득력 있게 표현하시면 좋겠습니다."
        },
        "비전과 성장 의지": {
            "title": "🔮 미래 비전과 기여 의지",
            "high": "개인의 성장과 조직의 발전을 균형있게 고려하는 장기적 비전을 가지고 계십니다. 회사에 대한 기여 의지도 명확합니다.",
            "low": "미래에 대한 계획은 있으나, 더 구체적이고 실현 가능한 비전을 제시하시면 좋겠습니다."
        }
    }
    
    feedback = feedback_map.get(category, {"title": category, "high": "", "low": ""})
    content = feedback["high"] if score >= 80 else feedback["low"]
    
    return feedback["title"], content

def export_results(answers: Dict, follow_up_answers: Dict, evaluation: Dict) -> str:
    """결과를 JSON 형태로 내보내기"""
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "company": "LIG넥스원",
        "interview_type": "조직적합도 인성면접 (임원 2차 면접용)",
        "answers": answers,
        "follow_up_answers": follow_up_answers,
        "evaluation": evaluation
    }
    
    return json.dumps(results, ensure_ascii=False, indent=2)

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="LIGNEX1 조직적합도 인성면접",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS 스타일링
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(45deg, #0066cc, #004499);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .company-info {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .question-container {
        background: #f1f8ff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    .score-display {
        background: linear-gradient(135deg, #0066cc, #004499);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .feedback-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 시스템 초기화
    interview_system = LIGNEXInterviewSystem()
    initialize_session_state()
    
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ LIGNEX1 조직적합도 인성면접 시스템</h1>
        <p>임원 2차 면접용 - 핵심가치 기반 평가</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바에 진행 상황 표시
    with st.sidebar:
        st.header("📊 면접 진행 상황")
        if st.session_state.interview_started:
            progress = (st.session_state.current_question + 1) / len(interview_system.questions)
            st.progress(progress)
            st.write(f"질문 {st.session_state.current_question + 1}/{len(interview_system.questions)}")
        else:
            st.write("면접을 시작해주세요.")
    
    # 메인 컨텐츠
    if not st.session_state.interview_started:
        # 시작 화면
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="company-info">
                <h3>🏢 LIG넥스원 기업 개요</h3>
                <p><strong>설립:</strong> 1976년 (금성정밀공업으로 시작)</p>
                <p><strong>사업분야:</strong> 대한민국 대표 방위산업체 - 정밀유도무기, 감시정찰체계, 지휘통신체계, 전자전체계</p>
                <p><strong>특징:</strong> 육해공 전 분야 통합 솔루션 제공, 전체 인력의 50%가 R&D 인력</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_open, col_positive = st.columns(2)
            with col_open:
                st.info("🔓 **OPEN**\n개방적 사고와 자세로 끊임없는 변화와 혁신을 실행")
            with col_positive:
                st.success("⚡ **POSITIVE**\n열정과 자신감으로 끝까지 목표를 달성")
        
        with col2:
            st.markdown("### 📋 면접 진행 방식")
            st.markdown("""
            - 총 5개의 조직적합도 관련 질문
            - 각 질문에 충분히 생각하고 답변
            - 답변에 따라 후속 질문 가능
            - 핵심가치(OPEN, POSITIVE) 중심 평가
            - 면접 완료 후 종합 평가 제공
            """)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 면접 시작하기", type="primary", use_container_width=True):
                st.session_state.interview_started = True
                st.rerun()
    
    elif not st.session_state.interview_completed:
        # 면접 진행 화면
        current_q = interview_system.questions[st.session_state.current_question]
        
        st.markdown(f"""
        <div class="question-container">
            <h4>질문 {st.session_state.current_question + 1}/5</h4>
            <p style="font-size: 1.1em; line-height: 1.6;">{current_q['text']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 답변 입력
        answer_key = f"answer_{st.session_state.current_question}"
        answer = st.text_area(
            "답변을 입력해 주세요:",
            height=150,
            key=answer_key,
            placeholder="구체적인 경험과 사례를 포함하여 답변해 주세요..."
        )
        
        # 후속 질문 처리
        if st.session_state.show_follow_up:
            st.markdown("### 💬 후속 질문")
            st.info(st.session_state.follow_up_question)
            
            follow_up_answer = st.text_area(
                "후속 질문 답변:",
                height=100,
                key=f"follow_up_{st.session_state.current_question}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("답변 완료", type="primary"):
                    if follow_up_answer.strip():
                        st.session_state.follow_up_answers[st.session_state.current_question] = {
                            "question": st.session_state.follow_up_question,
                            "answer": follow_up_answer
                        }
                    st.session_state.show_follow_up = False
                    st.rerun()
            
            with col2:
                if st.button("후속 질문 건너뛰기"):
                    st.session_state.show_follow_up = False
                    st.rerun()
        
        else:
            # 네비게이션 버튼
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.session_state.current_question > 0:
                    if st.button("⬅️ 이전 질문"):
                        st.session_state.current_question -= 1
                        st.rerun()
            
            with col3:
                if answer.strip():
                    if st.session_state.current_question < len(interview_system.questions) - 1:
                        if st.button("다음 질문 ➡️", type="primary"):
                            # 답변 저장
                            st.session_state.answers[st.session_state.current_question] = {
                                "question": current_q['text'],
                                "answer": answer,
                                "category": current_q['category'],
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            
                            # 후속 질문 확률적 표시 (30% 확률)
                            import random
                            if random.random() < 0.3 and current_q['follow_up']:
                                st.session_state.follow_up_question = random.choice(current_q['follow_up'])
                                st.session_state.show_follow_up = True
                                st.rerun()
                            else:
                                st.session_state.current_question += 1
                                st.rerun()
                    else:
                        if st.button("면접 완료 ✅", type="primary"):
                            # 마지막 답변 저장
                            st.session_state.answers[st.session_state.current_question] = {
                                "question": current_q['text'],
                                "answer": answer,
                                "category": current_q['category'],
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            st.session_state.interview_completed = True
                            st.rerun()
                else:
                    st.warning("답변을 입력해 주세요.")
    
    else:
        # 결과 화면
        st.markdown("## 📊 면접 결과 평가")
        
        # 점수 계산
        total_score = 0
        category_scores = {}
        
        for idx, answer_data in st.session_state.answers.items():
            score = evaluate_answer(
                answer_data['answer'], 
                answer_data['category'], 
                interview_system.keywords_by_category
            )
            
            # 후속 질문 보너스
            if idx in st.session_state.follow_up_answers:
                score += 5
            
            category_scores[answer_data['category']] = score
            total_score += score
        
        average_score = round(total_score / len(st.session_state.answers))
        grade = get_grade(average_score)
        
        # 점수 표시
        st.markdown(f"""
        <div class="score-display">
            <h2>{average_score}점 / 100점</h2>
            <h3>{grade}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 상세 피드백
        st.markdown("### 📋 상세 평가")
        
        for category, score in category_scores.items():
            title, content = generate_feedback(category, score)
            st.markdown(f"""
            <div class="feedback-item">
                <h4>{title}</h4>
                <p>{content}</p>
                <small>점수: {score}/100</small>
            </div>
            """, unsafe_allow_html=True)
        
        # 종합 평가
        st.markdown(f"""
        <div class="feedback-item">
            <h4>📊 종합 평가</h4>
            <p>전체적으로 {average_score}점의 {'우수한' if average_score >= 80 else '양호한' if average_score >= 70 else '보통의'} 답변을 보여주셨습니다. 
            LIG넥스원의 핵심가치에 대한 이해도가 {'높고' if average_score >= 80 else '적절하고'}, 
            방위산업에 대한 사명감도 {'잘 드러났습니다.' if average_score >= 80 else '확인할 수 있었습니다.'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if average_score < 80:
            st.markdown("""
            <div class="feedback-item">
                <h4>💡 개선 제안</h4>
                <p>더 구체적인 경험 사례와 수치적 성과를 포함하시면 더욱 설득력 있는 답변이 될 것입니다. 
                또한 LIG넥스원의 최신 기술 동향과 사업 영역에 대한 관심도 함께 어필하시기 바랍니다.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 결과 다운로드 및 재시작
        col1, col2 = st.columns(2)
        
        with col1:
            evaluation_data = {
                "total_score": average_score,
                "grade": grade,
                "category_scores": category_scores
            }
            
            results_json = export_results(
                st.session_state.answers,
                st.session_state.follow_up_answers,
                evaluation_data
            )
            
            st.download_button(
                label="📄 결과 다운로드 (JSON)",
                data=results_json,
                file_name=f"LIGNEX1_면접결과_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            if st.button("🔄 새로운 면접 시작", type="primary"):
                # 세션 상태 초기화
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()