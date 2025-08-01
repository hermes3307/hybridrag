# AI Coder - 매뉴얼 기반 전문 AI 코드 생성 시스템

## 개요
AI Coder는 **매뉴얼 학습 기능**을 핵심으로 하는 전문적인 AI 기반 코드 생성 시스템입니다. ALTIBASE 매뉴얼이나 사용자 정의 매뉴얼(PDF, Word 등)을 Vector DB로 학습하여, 일반 AI 도구가 알지 못하는 전문적이고 특화된 코드를 정확하게 생성합니다.

## 🔥 핵심 차별화 기능

### 📚 매뉴얼 학습 시스템 (핵심 기능)
- **다양한 문서 형식 지원**: PDF, Word, Markdown, Text, HTML 파일 자동 처리
- **Vector DB 기반 지식 저장**: ChromaDB를 활용한 매뉴얼 내용 임베딩 및 저장
- **지능적 문서 청킹**: 매뉴얼의 논리적 구조를 파악하여 의미 단위로 분할
- **ALTIBASE 전문 지원**: ALTIBASE SQL, API, 설정 매뉴얼 특화 처리
- **사용자 정의 매뉴얼**: 회사 내부 코딩 가이드, 프레임워크 문서 등 맞춤형 학습

### 🎯 매뉴얼 기반 코드 생성
- **문법 정확성 보장**: 매뉴얼에서 학습한 정확한 문법과 API 사용법 적용
- **컨텍스트 기반 검색**: 코드 생성 시 관련된 매뉴얼 섹션을 RAG로 참조
- **버전별 매뉴얼 관리**: 다양한 버전의 매뉴얼을 구분하여 관리
- **실시간 매뉴얼 참조**: 코드 생성 과정에서 실시간으로 매뉴얼 내용 검증

### 🔍 매뉴얼 기반 코드 검증
- **문법 검사**: 매뉴얼 기준으로 문법 오류 자동 탐지
- **API 호환성 검사**: 매뉴얼에 정의된 API 스펙과 비교하여 호환성 검증
- **베스트 프랙티스 적용**: 매뉴얼에서 추출한 권장 사항 자동 적용
- **deprecated 기능 경고**: 매뉴얼 기반으로 비권장 기능 사용 시 경고

### 💡 지능적 매뉴얼 분석
- **자동 인덱싱**: 매뉴얼의 목차, 섹션, 키워드를 자동으로 추출하여 인덱싱
- **예제 코드 추출**: 매뉴얼 내 예제 코드를 자동으로 식별하고 분류
- **관련성 점수**: 사용자 요청과 매뉴얼 내용 간의 관련성을 수치화
- **매뉴얼 업데이트 감지**: 매뉴얼 파일 변경 시 자동 재학습

## 시스템 구조

### 핵심 컴포넌트
1. **Manual Processor**: PDF, Word, HTML 등 매뉴얼 파일 파싱 및 전처리
2. **Vector DB Manager**: ChromaDB 기반 매뉴얼 임베딩 저장 및 검색
3. **RAG Engine**: 매뉴얼 기반 정보 검색 및 컨텍스트 구성
4. **AI Code Generator**: Claude API + 매뉴얼 컨텍스트 기반 코드 생성
5. **Syntax Validator**: 매뉴얼 기준 문법 및 API 검증 시스템
6. **Manual Manager**: 매뉴얼 버전 관리 및 업데이트 감지
7. **GUI Interface**: 매뉴얼 업로드 및 코드 생성 통합 인터페이스

### 매뉴얼 기반 코드 생성 플로우
```
매뉴얼 업로드 → 문서 파싱 → 청킹 & 임베딩 → Vector DB 저장
     ↓              ↓           ↓            ↓
사용자 요청 → 매뉴얼 검색 → RAG 컨텍스트 → AI 코드 생성 → 매뉴얼 기반 검증 → 결과 출력
     ↓              ↓           ↓            ↓            ↓            ↓
키워드 추출 → 관련성 점수 → 참조 섹션 → 문법 적용 → API 호환성 → 파일 저장
```

### ALTIBASE 특화 처리
```
ALTIBASE 매뉴얼 → SQL 문법 추출 → API 명세 파싱 → 설정 가이드 → 예제 코드
      ↓              ↓            ↓            ↓           ↓
버전별 분류 → 문법 규칙 DB → API 스펙 DB → 구성 템플릿 → 코드 스니펫 DB
```

## 설치 및 설정

### 시스템 요구사항
- Python 3.8 이상
- 최소 4GB RAM
- 인터넷 연결 (Claude API 통신용)

### 필요한 패키지
```bash
pip install -r requirements.txt
```

주요 의존성:
- `anthropic`: Claude AI API 클라이언트
- `chromadb`: Vector DB for 매뉴얼 임베딩 저장
- `PyPDF2` / `pdfplumber`: PDF 매뉴얼 파싱
- `python-docx`: Word 문서 처리
- `beautifulsoup4`: HTML 매뉴얼 파싱
- `sentence-transformers`: 텍스트 임베딩 생성
- `tkinter`: GUI 인터페이스
- `pathlib`: 파일 시스템 관리
- `watchdog`: 매뉴얼 파일 변경 감지

### API 키 설정
1. `.env` 파일을 프로젝트 루트에 생성
2. 다음 환경 변수를 설정:

```env
# Claude AI API 키 (필수)
CLAUDE_API_KEY=your_claude_api_key_here

# GitHub API 키 (선택, 레포지토리 분석용)
GITHUB_API_KEY=your_github_api_key_here

# OpenAI API 키 (선택, 추가 AI 모델용)
OPENAI_API_KEY=your_openai_api_key_here
```

## 사용법

### 1. 매뉴얼 업로드 및 학습

#### GUI 모드에서 매뉴얼 업로드
```bash
python gui.py
```
1. "매뉴얼 관리" 탭 선택
2. "매뉴얼 업로드" 버튼 클릭
3. PDF, Word, HTML 파일 선택
4. 매뉴얼 타입 설정 (ALTIBASE, 사용자 정의 등)
5. 자동 파싱 및 Vector DB 저장 확인

#### CLI 모드에서 매뉴얼 처리
```bash
# ALTIBASE 매뉴얼 업로드
python main.py --upload-manual ./manuals/altibase_sql_reference.pdf --type altibase

# 사용자 정의 매뉴얼 업로드
python main.py --upload-manual ./manuals/company_coding_guide.docx --type custom

# 매뉴얼 디렉토리 일괄 처리
python main.py --upload-directory ./manuals/ --auto-detect-type
```

### 2. 매뉴얼 기반 코드 생성

#### ALTIBASE 코드 생성 예시
```python
# ALTIBASE SQL 쿼리 생성
coder = AICoder()
coder.load_manuals(manual_type="altibase")

code = coder.generate_code(
    task="Create a stored procedure for user authentication",
    language="sql",
    manual_context="altibase",
    specifications="Include password hashing and session management"
)
```

#### 사용자 정의 매뉴얼 기반 코드 생성
```python
# 회사 내부 프레임워크 코드 생성
code = coder.generate_code_with_manual(
    task="Create API endpoint using company framework",
    manual_files=["./manuals/internal_api_guide.pdf"],
    language="python",
    specifications="Follow company security guidelines"
)
```

### 3. 매뉴얼 기반 코드 검증
```python
# 생성된 코드의 매뉴얼 준수성 검사
validation_result = coder.validate_code_against_manual(
    code=generated_code,
    manual_type="altibase",
    check_syntax=True,
    check_api_compatibility=True
)

print(f"문법 정확성: {validation_result.syntax_score}")
print(f"API 호환성: {validation_result.api_compatibility}")
print(f"개선 제안: {validation_result.suggestions}")
```

## GUI 사용 가이드

### 메인 화면 구성
1. **매뉴얼 관리 탭**: 매뉴얼 업로드, 파싱, Vector DB 관리
2. **코드 생성 탭**: 매뉴얼 기반 AI 코드 생성 및 편집
3. **코드 검증 탭**: 매뉴얼 기준 문법 및 API 검증
4. **매뉴얼 검색 탭**: Vector DB에서 매뉴얼 내용 검색 및 참조
5. **설정 탭**: API 키, 매뉴얼 설정, 고급 옵션

### 매뉴얼 관리 탭
- **업로드**: PDF, Word, HTML 매뉴얼 파일 업로드
- **파싱 상태**: 문서 파싱 및 청킹 진행 상황 표시
- **Vector DB 상태**: 임베딩 저장 및 인덱싱 상태 확인
- **매뉴얼 목록**: 업로드된 매뉴얼 목록 및 메타데이터
- **삭제/수정**: 매뉴얼 삭제 및 재처리

### 코드 생성 탭
- **매뉴얼 선택**: 참조할 매뉴얼 타입 선택 (ALTIBASE, 사용자 정의 등)
- **생성**: 매뉴얼 컨텍스트 기반 코드 생성
- **검증**: 실시간 매뉴얼 기준 문법 검사
- **참조 정보**: 사용된 매뉴얼 섹션 및 관련성 점수 표시
- **저장**: 생성된 코드와 참조 정보를 함께 저장

## 지원되는 매뉴얼 타입 및 언어

### 매뉴얼 문서 형식
- **PDF**: 일반적인 매뉴얼 및 기술 문서
- **Word (.docx)**: 편집 가능한 문서 형식
- **HTML**: 웹 기반 문서 및 온라인 매뉴얼
- **Markdown (.md)**: 개발자 문서 및 README
- **Text (.txt)**: 단순 텍스트 형식 매뉴얼

### ALTIBASE 전문 지원
- **SQL Reference Manual**: ALTIBASE SQL 문법 및 함수
- **API Reference**: ALTIBASE CLI, ODBC, JDBC API
- **Administrator Manual**: 설정 및 관리 가이드
- **Replication Manual**: 이중화 설정 및 관리
- **Error Message Reference**: 에러 코드 및 해결 방법

### 지원되는 프로그래밍 언어 (매뉴얼 기반)
- **SQL**: ALTIBASE SQL, Standard SQL
- **Python**: 사용자 정의 프레임워크 및 라이브러리
- **Java**: ALTIBASE JDBC, 사용자 정의 API
- **C/C++**: ALTIBASE CLI, 네이티브 라이브러리
- **JavaScript**: 사용자 정의 웹 프레임워크
- **기타**: 매뉴얼에 따라 모든 언어 지원 가능

## 고급 기능

### 매뉴얼 버전 관리
- **버전별 매뉴얼 분리**: 동일 제품의 여러 버전 매뉴얼 구분 관리
- **호환성 검사**: 코드와 매뉴얼 버전 간 호환성 자동 검증
- **마이그레이션 가이드**: 버전 업그레이드 시 변경사항 자동 안내

### 매뉴얼 기반 코드 품질 관리
- **매뉴얼 준수성 점수**: 생성된 코드의 매뉴얼 준수 정도를 수치화
- **베스트 프랙티스 적용**: 매뉴얼에서 추출한 권장 사항 자동 적용
- **안티패턴 탐지**: 매뉴얼에서 금지된 패턴 사용 시 경고

### 학습형 매뉴얼 시스템
- **사용 패턴 분석**: 자주 참조되는 매뉴얼 섹션 우선 순위 부여
- **컨텍스트 학습**: 특정 작업 타입에 대한 매뉴얼 매핑 최적화
- **피드백 학습**: 사용자 피드백을 통한 검색 정확도 개선

## API 레퍼런스

### AICoder 클래스 (매뉴얼 기반)
```python
class AICoder:
    def __init__(self, api_key: str, vector_db_path: str = "./manual_db")
    def upload_manual(self, file_path: str, manual_type: str, version: str = None) -> bool
    def generate_code_with_manual(self, task: str, language: str, manual_context: str, **kwargs) -> CodeResult
    def validate_code_against_manual(self, code: str, manual_type: str, **kwargs) -> ValidationResult
    def search_manual(self, query: str, manual_type: str = None, top_k: int = 5) -> SearchResult
    def get_manual_stats(self) -> Dict
```

### ManualProcessor 클래스
```python
class ManualProcessor:
    def parse_pdf(self, file_path: str) -> List[DocumentChunk]
    def parse_docx(self, file_path: str) -> List[DocumentChunk]
    def parse_html(self, file_path: str) -> List[DocumentChunk]
    def extract_code_examples(self, content: str) -> List[CodeExample]
    def chunk_document(self, content: str, chunk_size: int = 512) -> List[str]
```

### 주요 메서드
- `upload_manual()`: 매뉴얼 파일 업로드 및 Vector DB 저장
- `generate_code_with_manual()`: 매뉴얼 컨텍스트 기반 코드 생성
- `validate_code_against_manual()`: 매뉴얼 기준 코드 검증
- `search_manual()`: Vector DB에서 매뉴얼 내용 검색

## 예제 코드

### ALTIBASE 매뉴얼 기반 SQL 생성
```python
from ai_coder import AICoder

# AI Coder 초기화
coder = AICoder(api_key="your_claude_api_key", vector_db_path="./altibase_manuals")

# ALTIBASE 매뉴얼 업로드
coder.upload_manual(
    file_path="./manuals/ALTIBASE_SQL_Reference.pdf",
    manual_type="altibase",
    version="7.1"
)

# ALTIBASE SQL 코드 생성
sql_code = coder.generate_code_with_manual(
    task="Create a stored procedure for batch data processing with error handling",
    language="sql",
    manual_context="altibase",
    specifications="Include transaction management and logging"
)

print("Generated SQL Code:")
print(sql_code.code)
print(f"\nManual References Used: {len(sql_code.references)}")
for ref in sql_code.references:
    print(f"- {ref.section}: {ref.relevance_score}")
```

### 매뉴얼 기반 코드 검증
```python
# 생성된 코드의 매뉴얼 준수성 검사
validation_result = coder.validate_code_against_manual(
    code=sql_code.code,
    manual_type="altibase",
    check_syntax=True,
    check_api_compatibility=True,
    check_best_practices=True
)

print(f"매뉴얼 준수성 점수: {validation_result.compliance_score}/100")
print(f"문법 정확성: {validation_result.syntax_score}/10")
print(f"API 호환성: {validation_result.api_compatibility}/10")

if validation_result.suggestions:
    print("\n개선 제안:")
    for suggestion in validation_result.suggestions:
        print(f"- {suggestion.type}: {suggestion.description}")
        print(f"  참조 매뉴얼: {suggestion.manual_reference}")
```

### 사용자 정의 매뉴얼 활용
```python
# 회사 내부 코딩 가이드 업로드
coder.upload_manual(
    file_path="./manuals/company_python_guide.docx",
    manual_type="custom_python",
    version="2024"
)

# 회사 가이드에 따른 코드 생성
python_code = coder.generate_code_with_manual(
    task="Create a data processing pipeline with company standards",
    language="python",
    manual_context="custom_python",
    specifications="Follow company security and logging standards"
)

print(python_code.code)
```

## 문제 해결

### 매뉴얼 관련 문제
1. **매뉴얼 파싱 실패**: 
   - PDF 암호화 확인 (암호화된 PDF는 먼저 해제 필요)
   - 파일 권한 확인 (읽기 권한 필요)
   - 지원되지 않는 문서 형식인지 확인

2. **Vector DB 저장 실패**:
   - 디스크 공간 부족 확인
   - ChromaDB 버전 호환성 확인
   - 임베딩 모델 다운로드 상태 확인

3. **매뉴얼 검색 정확도 저하**:
   - 문서 청킹 크기 조절 (기본값: 512 토큰)
   - 임베딩 모델 변경 고려
   - 매뉴얼 전처리 품질 확인

### 성능 최적화
1. **대용량 매뉴얼 처리**:
   - 배치 처리 활용 (--batch-size 옵션)
   - 멀티프로세싱 활성화 (--parallel-processing)
   - 청킹 크기 최적화

2. **검색 속도 개선**:
   - Vector DB 인덱스 최적화
   - 캐시 활용 설정
   - 불필요한 매뉴얼 삭제

### 로그 확인
```bash
# 매뉴얼 처리 상세 로그
export AI_CODER_LOG_LEVEL=DEBUG
export MANUAL_PROCESSOR_VERBOSE=true
python gui.py
```

로그 파일 위치:
- 일반 로그: `logs/ai_coder.log`
- 매뉴얼 처리 로그: `logs/manual_processor.log`
- Vector DB 로그: `logs/vector_db.log`

## 성능 벤치마크

### 매뉴얼 처리 성능
- **PDF 파싱**: 100페이지 기준 평균 30-60초
- **Word 문서**: 50페이지 기준 평균 15-30초
- **HTML 문서**: 크기에 관계없이 평균 5-15초
- **Vector DB 저장**: 1000개 청크 기준 평균 2-5분

### 코드 생성 속도 (매뉴얼 기반)
- **간단한 함수**: 매뉴얼 검색 + 생성 평균 5-8초
- **복잡한 클래스**: 매뉴얼 검색 + 생성 평균 15-25초
- **전체 모듈**: 매뉴얼 검색 + 생성 평균 30-60초

### 지원되는 매뉴얼 크기
- **단일 PDF**: 최대 500MB (약 1000페이지)
- **Vector DB**: 최대 10GB (수백 개 매뉴얼)
- **동시 업로드**: 최대 10개 파일

## 보안 고려사항

### 매뉴얼 데이터 보안
- **로컬 저장**: 모든 매뉴얼은 로컬 Vector DB에만 저장
- **암호화**: 민감한 매뉴얼의 경우 AES-256 암호화 지원
- **접근 제어**: 매뉴얼별 사용자 권한 관리
- **데이터 무결성**: 매뉴얼 변조 탐지 및 체크섬 검증

### API 보안
- **키 관리**: API 키는 환경 변수로만 관리
- **HTTPS 통신**: Claude API와의 모든 통신 암호화
- **민감 정보 필터링**: 코드 생성 시 민감 정보 자동 마스킹
- **감사 로그**: 모든 API 호출 및 매뉴얼 접근 기록

### 생성 코드 안전성
- **매뉴얼 기반 검증**: 생성된 코드의 보안 패턴 검사
- **취약점 탐지**: OWASP 기준 보안 취약점 자동 탐지
- **안전한 패턴 강제**: 매뉴얼에 정의된 보안 가이드라인 자동 적용

## 라이선스
MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

## 기여 가이드

### 개발 환경 설정
1. 저장소 포크
2. 개발 브랜치 생성
3. 변경사항 구현
4. 테스트 실행
5. Pull Request 제출

### 코드 스타일
- PEP 8 (Python)
- ESLint (JavaScript)
- 커밋 메시지는 영어로 작성

## 지원 및 문의

- **이슈 트래커**: GitHub Issues
- **문서**: [프로젝트 위키](https://github.com/your-repo/ai-coder/wiki)
- **이메일**: support@ai-coder.com

## 버전 히스토리

### v1.0.0 (2025-01-01)
- **매뉴얼 학습 시스템** 초기 릴리스
- PDF, Word, HTML 매뉴얼 파싱 지원
- ChromaDB 기반 Vector DB 구현
- ALTIBASE 매뉴얼 특화 지원
- 기본 코드 생성 및 검증 기능

### v1.1.0 (예정)
- **다중 매뉴얼 버전 관리** 기능 추가
- 매뉴얼 기반 **자동 마이그레이션 가이드**
- **학습형 검색 시스템** 도입 (사용 패턴 기반 최적화)
- **배치 매뉴얼 처리** 성능 개선
- **플러그인 시스템**: 사용자 정의 매뉴얼 파서 지원

### v1.2.0 (계획)
- **멀티모달 매뉴얼 지원**: 이미지, 다이어그램 포함 문서 처리
- **실시간 매뉴얼 동기화**: 온라인 매뉴얼 자동 업데이트
- **팀 협업 기능**: 매뉴얼 공유 및 버전 관리
- **고급 코드 분석**: 매뉴얼 기반 성능 및 보안 최적화

---

## 🎯 AI Coder의 핵심 가치

**"매뉴얼이 곧 코딩 표준"** - 일반 AI 도구와 차별화되는 전문성
- ✅ **정확한 문법**: 매뉴얼 기반 100% 정확한 API 사용법
- ✅ **버전 호환성**: 특정 버전 매뉴얼에 맞는 코드 생성
- ✅ **전문 영역**: ALTIBASE 같은 전문 시스템 완벽 지원
- ✅ **기업 표준**: 회사 내부 가이드라인 자동 준수

**AI Coder**로 매뉴얼 기반의 정확하고 전문적인 개발을 경험하세요! 📚🚀