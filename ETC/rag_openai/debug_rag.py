#!/usr/bin/env python3
"""
🔍 RAG Process Debugger
Detailed analysis and visualization of RAG (Retrieval Augmented Generation) process
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import re
from datetime import datetime

# Import RAG components
from vector_manager import VectorStoreManager
from query_engine import ConversationalQueryEngine


@dataclass
class RAGStep:
    """🔍 Individual step in RAG process"""
    step_number: int
    step_name: str
    description: str
    input_data: Any
    output_data: Any
    processing_time: float
    status: str  # success, failed, warning
    details: Dict[str, Any]
    timestamp: str


@dataclass
class RAGDebugResult:
    """📊 Complete RAG debugging result"""
    question: str
    total_processing_time: float
    steps: List[RAGStep]
    final_result: str
    success: bool
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = None


class RAGProcessDebugger:
    """🔍 Detailed RAG Process Debugger"""
    
    def __init__(self, vector_manager: VectorStoreManager, query_engine: ConversationalQueryEngine):
        self.vector_manager = vector_manager
        self.query_engine = query_engine
        self.debug_results = []
        
        # Domain keywords for analysis
        self.domain_keywords = {
            'database': ['database', 'db', 'sql', 'table', 'index', 'query', 'schema'],
            'performance': ['performance', 'optimization', 'tuning', 'slow', 'fast', 'memory'],
            'altibase': ['altibase', 'ipcda', 'apre', 'isql', 'iloader', 'hybrid'],
            'network': ['network', 'connection', 'port', 'protocol', 'tcp', 'ip'],
            'configuration': ['config', 'setting', 'parameter', 'setup', 'installation']
        }
        
        # Question type patterns
        self.question_patterns = {
            'what_is': [r'what\s+is\s+(.+)', r'뭐야', r'무엇', r'정의'],
            'how_to': [r'how\s+to\s+(.+)', r'어떻게', r'방법'],
            'troubleshoot': [r'error|problem|issue|fix', r'오류', r'문제', r'해결'],
            'comparison': [r'difference|compare|vs', r'차이', r'비교']
        }

    async def debug_rag_process(self, question: str, show_details: bool = True) -> RAGDebugResult:
        """🔍 RAG 프로세스 전체를 디버깅"""
        
        print(f"\n🔍 {'='*60}")
        print(f"🧠 RAG 프로세스 디버깅 시작")
        print(f"❓ 질문: '{question}'")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 {'='*60}\n")
        
        start_time = time.time()
        steps = []
        
        try:
            # Step 1: 질문 분석
            step1_result = await self._debug_step1_question_analysis(question)
            steps.append(step1_result)
            if show_details: self._print_step_details(step1_result)
            
            # Step 2: 키워드 추출
            step2_result = await self._debug_step2_keyword_extraction(question, step1_result.output_data)
            steps.append(step2_result)
            if show_details: self._print_step_details(step2_result)
            
            # Step 3: 검색 쿼리 생성
            step3_result = await self._debug_step3_search_query_generation(step2_result.output_data)
            steps.append(step3_result)
            if show_details: self._print_step_details(step3_result)
            
            # Step 4: 벡터 검색
            step4_result = await self._debug_step4_vector_search(step3_result.output_data)
            steps.append(step4_result)
            if show_details: self._print_step_details(step4_result)
            
            # Step 5: 결과 필터링 및 랭킹
            step5_result = await self._debug_step5_result_filtering(step4_result.output_data)
            steps.append(step5_result)
            if show_details: self._print_step_details(step5_result)
            
            # Step 6: 컨텍스트 구성
            step6_result = await self._debug_step6_context_building(step5_result.output_data)
            steps.append(step6_result)
            if show_details: self._print_step_details(step6_result)
            
            # Step 7: LLM 프롬프트 생성
            step7_result = await self._debug_step7_prompt_generation(question, step6_result.output_data, step1_result.output_data)
            steps.append(step7_result)
            if show_details: self._print_step_details(step7_result)
            
            # Step 8: 최종 응답 생성 (시뮬레이션)
            step8_result = await self._debug_step8_response_generation(step7_result.output_data)
            steps.append(step8_result)
            if show_details: self._print_step_details(step8_result)
            
            total_time = time.time() - start_time
            
            # 성능 메트릭 계산
            performance_metrics = self._calculate_performance_metrics(steps, total_time)
            
            debug_result = RAGDebugResult(
                question=question,
                total_processing_time=total_time,
                steps=steps,
                final_result=step8_result.output_data.get('final_answer', 'No answer generated'),
                success=True,
                performance_metrics=performance_metrics
            )
            
            # 결과 저장
            self.debug_results.append(debug_result)
            
            # 최종 요약 출력
            if show_details:
                self._print_final_summary(debug_result)
            
            return debug_result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_result = RAGDebugResult(
                question=question,
                total_processing_time=total_time,
                steps=steps,
                final_result="Error occurred during processing",
                success=False,
                error_message=str(e)
            )
            
            print(f"❌ RAG 프로세스 오류: {e}")
            return error_result

    async def _debug_step1_question_analysis(self, question: str) -> RAGStep:
        """Step 1: 질문 분석"""
        start_time = time.time()
        
        # 질문 타입 감지
        question_type = 'general'
        detected_pattern = None
        
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    question_type = q_type
                    detected_pattern = pattern
                    break
            if detected_pattern:
                break
        
        # 질문 복잡도 분석
        complexity_score = self._calculate_question_complexity(question)
        
        # 언어 감지
        language = 'korean' if any(ord(char) > 127 for char in question) else 'english'
        
        processing_time = time.time() - start_time
        
        output_data = {
            'question_type': question_type,
            'detected_pattern': detected_pattern,
            'complexity_score': complexity_score,
            'language': language,
            'word_count': len(question.split()),
            'character_count': len(question)
        }
        
        return RAGStep(
            step_number=1,
            step_name="질문 분석 (Question Analysis)",
            description="사용자 질문의 타입, 복잡도, 언어를 분석합니다",
            input_data={'question': question},
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'patterns_checked': len(self.question_patterns),
                'complexity_factors': ['word_count', 'question_marks', 'technical_terms']
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step2_keyword_extraction(self, question: str, question_analysis: Dict) -> RAGStep:
        """Step 2: 키워드 추출"""
        start_time = time.time()
        
        # 간단한 키워드 추출 (실제로는 NLTK 사용)
        words = question.lower().split()
        
        # 불용어 제거
        stop_words = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', '가', '는', '을', '를', '이', '그', '그것', '뭐야', '무엇'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 기술 용어 추출
        technical_terms = []
        for word in words:
            for domain, domain_words in self.domain_keywords.items():
                if word.lower() in domain_words:
                    technical_terms.append(word.lower())
        
        # 도메인 식별
        identified_domains = []
        for domain, domain_words in self.domain_keywords.items():
            if any(word in keywords + technical_terms for word in domain_words):
                identified_domains.append(domain)
        
        processing_time = time.time() - start_time
        
        output_data = {
            'keywords': keywords,
            'technical_terms': list(set(technical_terms)),
            'identified_domains': identified_domains,
            'keyword_count': len(keywords),
            'technical_term_count': len(set(technical_terms))
        }
        
        return RAGStep(
            step_number=2,
            step_name="키워드 추출 (Keyword Extraction)",
            description="질문에서 중요한 키워드와 기술 용어를 추출합니다",
            input_data={'question': question, 'analysis': question_analysis},
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'stop_words_removed': len([w for w in words if w in stop_words]),
                'domains_checked': len(self.domain_keywords)
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step3_search_query_generation(self, keyword_data: Dict) -> RAGStep:
        """Step 3: 검색 쿼리 생성"""
        start_time = time.time()
        
        keywords = keyword_data['keywords']
        technical_terms = keyword_data['technical_terms']
        domains = keyword_data['identified_domains']
        
        # 다양한 검색 쿼리 생성
        search_queries = []
        
        # 1. 메인 키워드 쿼리
        if keywords:
            main_query = ' '.join(keywords[:3])  # 상위 3개 키워드
            search_queries.append({
                'query': main_query,
                'type': 'main_keywords',
                'priority': 1
            })
        
        # 2. 기술 용어 쿼리
        if technical_terms:
            tech_query = ' '.join(technical_terms)
            search_queries.append({
                'query': tech_query,
                'type': 'technical_terms',
                'priority': 2
            })
        
        # 3. 도메인 특화 쿼리
        for domain in domains:
            domain_query = f"{domain} {' '.join(keywords[:2])}"
            search_queries.append({
                'query': domain_query,
                'type': f'domain_{domain}',
                'priority': 3
            })
        
        # 중복 제거
        unique_queries = []
        seen_queries = set()
        for query_info in search_queries:
            if query_info['query'] not in seen_queries:
                unique_queries.append(query_info)
                seen_queries.add(query_info['query'])
        
        processing_time = time.time() - start_time
        
        output_data = {
            'search_queries': unique_queries,
            'query_count': len(unique_queries),
            'query_types': list(set(q['type'] for q in unique_queries))
        }
        
        return RAGStep(
            step_number=3,
            step_name="검색 쿼리 생성 (Search Query Generation)",
            description="키워드를 바탕으로 벡터 검색용 쿼리들을 생성합니다",
            input_data=keyword_data,
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'original_queries': len(search_queries),
                'duplicates_removed': len(search_queries) - len(unique_queries)
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step4_vector_search(self, query_data: Dict) -> RAGStep:
        """Step 4: 벡터 검색"""
        start_time = time.time()
        
        search_queries = query_data['search_queries']
        all_results = []
        search_details = []
        
        # 각 쿼리로 검색 수행
        for query_info in search_queries:
            query = query_info['query']
            try:
                # 실제 벡터 검색 수행
                if self.vector_manager:
                    results = await self.vector_manager.search(query, k=5)
                    all_results.extend(results)
                    
                    search_details.append({
                        'query': query,
                        'type': query_info['type'],
                        'results_count': len(results),
                        'top_score': results[0]['score'] if results else 0,
                        'status': 'success'
                    })
                else:
                    # 벡터 매니저가 없는 경우 시뮬레이션
                    simulated_results = self._simulate_search_results(query, 3)
                    all_results.extend(simulated_results)
                    
                    search_details.append({
                        'query': query,
                        'type': query_info['type'],
                        'results_count': len(simulated_results),
                        'top_score': simulated_results[0]['score'] if simulated_results else 0,
                        'status': 'simulated'
                    })
                    
            except Exception as e:
                search_details.append({
                    'query': query,
                    'type': query_info['type'],
                    'results_count': 0,
                    'error': str(e),
                    'status': 'failed'
                })
        
        processing_time = time.time() - start_time
        
        # 결과 통계
        total_results = len(all_results)
        successful_queries = len([d for d in search_details if d['status'] in ['success', 'simulated']])
        
        output_data = {
            'search_results': all_results,
            'search_details': search_details,
            'total_results': total_results,
            'successful_queries': successful_queries,
            'average_score': sum(r['score'] for r in all_results) / total_results if total_results > 0 else 0
        }
        
        return RAGStep(
            step_number=4,
            step_name="벡터 검색 (Vector Search)",
            description="생성된 쿼리로 벡터 데이터베이스에서 관련 문서를 검색합니다",
            input_data=query_data,
            output_data=output_data,
            processing_time=processing_time,
            status='success' if successful_queries > 0 else 'warning',
            details={
                'queries_executed': len(search_queries),
                'vector_db_available': self.vector_manager is not None
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step5_result_filtering(self, search_data: Dict) -> RAGStep:
        """Step 5: 결과 필터링 및 랭킹"""
        start_time = time.time()
        
        all_results = search_data['search_results']
        
        # 중복 제거
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content = result.get('payload', {}).get('text', '')[:200]
            content_hash = hash(content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # 점수별 정렬
        sorted_results = sorted(unique_results, key=lambda x: x.get('score', 0), reverse=True)
        
        # 상위 결과 선택 (최대 8개)
        top_results = sorted_results[:8]
        
        # 필터링 통계
        duplicates_removed = len(all_results) - len(unique_results)
        score_threshold = 0.3  # 임계점수
        high_quality_results = [r for r in top_results if r.get('score', 0) > score_threshold]
        
        processing_time = time.time() - start_time
        
        output_data = {
            'filtered_results': top_results,
            'high_quality_results': high_quality_results,
            'total_after_filtering': len(top_results),
            'duplicates_removed': duplicates_removed,
            'score_distribution': {
                'min': min(r['score'] for r in top_results) if top_results else 0,
                'max': max(r['score'] for r in top_results) if top_results else 0,
                'avg': sum(r['score'] for r in top_results) / len(top_results) if top_results else 0
            }
        }
        
        return RAGStep(
            step_number=5,
            step_name="결과 필터링 (Result Filtering)",
            description="검색 결과에서 중복을 제거하고 관련성 점수로 정렬합니다",
            input_data={'original_count': len(all_results)},
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'score_threshold': score_threshold,
                'max_results': 8
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step6_context_building(self, filtered_data: Dict) -> RAGStep:
        """Step 6: 컨텍스트 구성"""
        start_time = time.time()
        
        top_results = filtered_data['filtered_results']
        max_context_length = 2000  # 최대 컨텍스트 길이
        
        context_parts = []
        current_length = 0
        sources_used = []
        
        for i, result in enumerate(top_results, 1):
            payload = result.get('payload', {})
            text = payload.get('text', '').strip()
            source = payload.get('source_file', 'Unknown')
            
            if text and current_length < max_context_length:
                # 소스 정보와 함께 텍스트 추가
                context_part = f"[Source {i}: {source}]\n{text}\n"
                
                if current_length + len(context_part) <= max_context_length:
                    context_parts.append(context_part)
                    current_length += len(context_part)
                    sources_used.append(source)
                else:
                    # 잘라서 추가
                    remaining_space = max_context_length - current_length - 100
                    if remaining_space > 100:
                        truncated_text = text[:remaining_space] + "..."
                        context_parts.append(f"[Source {i}: {source}]\n{truncated_text}\n")
                        sources_used.append(source)
                    break
        
        final_context = "\n".join(context_parts)
        
        processing_time = time.time() - start_time
        
        output_data = {
            'final_context': final_context,
            'context_length': len(final_context),
            'sources_used': sources_used,
            'sources_count': len(set(sources_used)),
            'truncated': len(final_context) >= max_context_length - 100
        }
        
        return RAGStep(
            step_number=6,
            step_name="컨텍스트 구성 (Context Building)",
            description="필터링된 결과들을 하나의 컨텍스트로 조합합니다",
            input_data={'available_results': len(top_results)},
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'max_length': max_context_length,
                'parts_included': len(context_parts)
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step7_prompt_generation(self, question: str, context_data: Dict, question_analysis: Dict) -> RAGStep:
        """Step 7: LLM 프롬프트 생성"""
        start_time = time.time()
        
        context = context_data['final_context']
        question_type = question_analysis['question_type']
        
        # 질문 타입별 프롬프트 템플릿
        prompt_templates = {
            'what_is': """다음 문서 컨텍스트를 바탕으로 {topic}이 무엇인지 상세히 설명해주세요:

컨텍스트:
{context}

질문: {question}

다음 사항을 포함하여 답변해주세요:
1. 정의 및 개요
2. 주요 특징
3. 사용 사례
4. 중요한 기술적 세부사항

답변:""",
            
            'how_to': """다음 문서 컨텍스트를 바탕으로 {topic} 방법을 단계별로 설명해주세요:

컨텍스트:
{context}

질문: {question}

다음 사항을 포함하여 답변해주세요:
1. 단계별 지침
2. 전제조건
3. 주의사항
4. 예상 결과

답변:""",
            
            'general': """다음 문서 컨텍스트를 바탕으로 질문에 답변해주세요:

컨텍스트:
{context}

질문: {question}

위 문서 내용을 바탕으로 포괄적인 답변을 제공해주세요.

답변:"""
        }
        
        # 프롬프트 선택 및 생성
        template = prompt_templates.get(question_type, prompt_templates['general'])
        
        # 주제 추출 (간단한 방법)
        topic = question.replace('뭐야', '').replace('what is', '').strip()
        
        final_prompt = template.format(
            topic=topic,
            context=context,
            question=question
        )
        
        processing_time = time.time() - start_time
        
        output_data = {
            'final_prompt': final_prompt,
            'prompt_length': len(final_prompt),
            'template_used': question_type,
            'topic_extracted': topic,
            'context_included': len(context) > 0
        }
        
        return RAGStep(
            step_number=7,
            step_name="프롬프트 생성 (Prompt Generation)",
            description="LLM을 위한 최적화된 프롬프트를 생성합니다",
            input_data={'question': question, 'question_type': question_type},
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'templates_available': len(prompt_templates),
                'context_characters': len(context)
            },
            timestamp=datetime.now().isoformat()
        )

    async def _debug_step8_response_generation(self, prompt_data: Dict) -> RAGStep:
        """Step 8: 응답 생성 (시뮬레이션)"""
        start_time = time.time()
        
        prompt = prompt_data['final_prompt']
        
        # OpenAI API 호출 시뮬레이션
        # 실제로는 OpenAI API를 호출하지만, 여기서는 시뮬레이션
        
        simulated_response = """🎯 **IPCDA (In-Process Communication Direct Access)에 대해 설명드리겠습니다!** 

📖 **정의 및 개요:**
IPCDA는 Altibase에서 제공하는 고성능 통신 방식으로, 메모리 데이터베이스와 디스크 데이터베이스 간의 직접적인 통신을 가능하게 합니다.

🚀 **주요 특징:**
• 메모리 간 직접 접근으로 초고속 처리 ⚡
• 네트워크 오버헤드 최소화 🔄
• 대용량 데이터 처리에 최적화 📊

⚙️ **설정 방법:**
altibase.properties 파일에서 IPCDA_CHANNEL_COUNT와 IPCDA_DATABLOCK_SIZE 등을 설정하여 사용할 수 있습니다.

이 정보는 제공된 문서 컨텍스트를 바탕으로 생성되었습니다."""
        
        # 응답 품질 메트릭 계산
        response_metrics = {
            'length': len(simulated_response),
            'contains_korean': any(ord(char) > 127 for char in simulated_response),
            'contains_emojis': any(char in '🎯📖🚀⚡🔄📊⚙️' for char in simulated_response),
            'structured': '**' in simulated_response and '•' in simulated_response
        }
        
        processing_time = time.time() - start_time
        
        output_data = {
            'final_answer': simulated_response,
            'response_metrics': response_metrics,
            'generation_method': 'simulated',  # 실제로는 'openai_api'
            'tokens_estimated': len(simulated_response) // 4  # 대략적인 토큰 수
        }
        
        return RAGStep(
            step_number=8,
            step_name="응답 생성 (Response Generation)",
            description="LLM이 컨텍스트를 바탕으로 최종 응답을 생성합니다",
            input_data={'prompt_length': len(prompt)},
            output_data=output_data,
            processing_time=processing_time,
            status='success',
            details={
                'model_used': 'gpt-4o-mini (simulated)',
                'temperature': 0.3
            },
            timestamp=datetime.now().isoformat()
        )

    def _calculate_question_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        complexity = 0.0
        
        # 단어 수
        word_count = len(question.split())
        complexity += min(word_count / 10, 1.0)
        
        # 질문 표시 수
        question_marks = question.count('?') + question.count('？')
        complexity += min(question_marks / 3, 0.5)
        
        # 기술 용어 포함 여부
        technical_terms = sum(1 for domain_words in self.domain_keywords.values() 
                             for word in domain_words if word in question.lower())
        complexity += min(technical_terms / 5, 0.5)
        
        return min(complexity, 1.0)

    def _simulate_search_results(self, query: str, count: int) -> List[Dict]:
        """검색 결과 시뮬레이션"""
        simulated_results = []
        
        for i in range(count):
            score = max(0.3, 1.0 - (i * 0.15))  # 점수가 감소하는 패턴
            
            result = {
                'score': score,
                'payload': {
                    'text': f"이것은 '{query}' 검색어에 대한 시뮬레이션된 문서 내용입니다. 문서 {i+1}번에서 관련 정보를 찾았습니다. IPCDA는 Altibase의 고성능 통신 기술로, 메모리 데이터베이스와 디스크 데이터베이스 간의 직접 통신을 지원합니다.",
                    'source_file': f'altibase_manual_part{i+1}.pdf',
                    'chunk_id': f'chunk_{i+1}_{hash(query) % 1000}'
                }
            }
            
            simulated_results.append(result)
        
        return simulated_results

    def _calculate_performance_metrics(self, steps: List[RAGStep], total_time: float) -> Dict[str, Any]:
        """성능 메트릭 계산"""
        
        step_times = [step.processing_time for step in steps]
        
        return {
            'total_time': total_time,
            'step_times': {step.step_name: step.processing_time for step in steps},
            'slowest_step': max(steps, key=lambda x: x.processing_time).step_name if steps else None,
            'fastest_step': min(steps, key=lambda x: x.processing_time).step_name if steps else None,
            'average_step_time': sum(step_times) / len(step_times) if step_times else 0,
            'successful_steps': len([s for s in steps if s.status == 'success']),
            'total_steps': len(steps)
        }

    def _print_step_details(self, step: RAGStep):
        """단계별 상세 정보 출력"""
        
        status_emoji = {
            'success': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }
        
        print(f"\n{status_emoji.get(step.status, '🔄')} **Step {step.step_number}: {step.step_name}**")
        print(f"📝 설명: {step.description}")
        print(f"⏱️ 처리 시간: {step.processing_time:.3f}초")
        
        # 입력 데이터 요약
        if step.input_data:
            print(f"📥 입력: {self._summarize_data(step.input_data)}")
        
        # 출력 데이터 요약
        if step.output_data:
            print(f"📤 출력: {self._summarize_data(step.output_data)}")
        
        # 상세 정보
        if step.details:
            print(f"🔍 세부사항: {step.details}")
        
        print("-" * 60)

    def _summarize_data(self, data: Any) -> str:
        """데이터 요약"""
        if isinstance(data, dict):
            summary_parts = []
            for key, value in data.items():
                if isinstance(value, list):
                    summary_parts.append(f"{key}({len(value)}개)")
                elif isinstance(value, str) and len(value) > 50:
                    summary_parts.append(f"{key}({len(value)}자)")
                else:
                    summary_parts.append(f"{key}={value}")
            return ", ".join(summary_parts)
        elif isinstance(data, list):
            return f"리스트({len(data)}개 항목)"
        else:
            return str(data)

    def _print_final_summary(self, debug_result: RAGDebugResult):
        """최종 요약 출력"""
        
        print(f"\n🎉 {'='*60}")
        print(f"🏁 RAG 프로세스 완료!")
        print(f"{'='*60}")
        
        print(f"❓ 원본 질문: {debug_result.question}")
        print(f"⏱️ 총 처리 시간: {debug_result.total_processing_time:.3f}초")
        print(f"✅ 성공 여부: {'성공' if debug_result.success else '실패'}")
        
        if debug_result.performance_metrics:
            metrics = debug_result.performance_metrics
            print(f"\n📊 **성능 메트릭:**")
            print(f"   • 가장 느린 단계: {metrics['slowest_step']}")
            print(f"   • 가장 빠른 단계: {metrics['fastest_step']}")
            print(f"   • 평균 단계 시간: {metrics['average_step_time']:.3f}초")
            print(f"   • 성공한 단계: {metrics['successful_steps']}/{metrics['total_steps']}")
        
        print(f"\n🎯 **최종 답변:**")
        print(debug_result.final_result)
        
        print(f"\n🔍 {'='*60}")

    def save_debug_result(self, debug_result: RAGDebugResult, filename: str = None):
        """디버그 결과를 JSON 파일로 저장"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_debug_{timestamp}.json"
        
        # dataclass를 dict로 변환
        result_dict = asdict(debug_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 디버그 결과가 {filename}에 저장되었습니다.")

    def get_debug_history(self) -> List[RAGDebugResult]:
        """디버그 히스토리 반환"""
        return self.debug_results

    async def compare_queries(self, questions: List[str]) -> Dict[str, Any]:
        """여러 질문의 RAG 프로세스 비교"""
        
        print(f"\n🔍 여러 질문 RAG 프로세스 비교 시작")
        print(f"📝 질문 수: {len(questions)}")
        print("-" * 60)
        
        comparison_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔸 질문 {i}: {question}")
            result = await self.debug_rag_process(question, show_details=False)
            comparison_results.append(result)
        
        # 비교 분석
        comparison_analysis = {
            'questions': questions,
            'results': comparison_results,
            'performance_comparison': {
                'fastest_question': min(comparison_results, key=lambda x: x.total_processing_time).question,
                'slowest_question': max(comparison_results, key=lambda x: x.total_processing_time).question,
                'average_time': sum(r.total_processing_time for r in comparison_results) / len(comparison_results),
                'success_rate': len([r for r in comparison_results if r.success]) / len(comparison_results)
            }
        }
        
        # 비교 결과 출력
        print(f"\n📊 **비교 분석 결과:**")
        print(f"   • 가장 빠른 질문: {comparison_analysis['performance_comparison']['fastest_question']}")
        print(f"   • 가장 느린 질문: {comparison_analysis['performance_comparison']['slowest_question']}")
        print(f"   • 평균 처리 시간: {comparison_analysis['performance_comparison']['average_time']:.3f}초")
        print(f"   • 성공률: {comparison_analysis['performance_comparison']['success_rate']:.1%}")
        
        return comparison_analysis


# 편의 함수들
async def debug_single_question(question: str, vector_manager=None, query_engine=None):
    """단일 질문 RAG 디버깅"""
    debugger = RAGProcessDebugger(vector_manager, query_engine)
    return await debugger.debug_rag_process(question)

async def debug_multiple_questions(questions: List[str], vector_manager=None, query_engine=None):
    """여러 질문 RAG 비교 디버깅"""
    debugger = RAGProcessDebugger(vector_manager, query_engine)
    return await debugger.compare_queries(questions)

async def quick_rag_demo():
    """RAG 프로세스 빠른 데모"""
    print("🚀 RAG 프로세스 빠른 데모 시작!")
    
    demo_questions = [
        "IPCDA가 뭐야?",
        "How to configure database performance?",
        "Altibase 설치 방법은?"
    ]
    
    debugger = RAGProcessDebugger(None, None)  # 시뮬레이션 모드
    
    for question in demo_questions:
        print(f"\n{'🔍 ' + '='*50}")
        await debugger.debug_rag_process(question, show_details=True)


if __name__ == "__main__":
    # 단독 실행시 데모 실행
    print("🔍 RAG 프로세스 디버거 - 독립 실행 모드")
    asyncio.run(quick_rag_demo())