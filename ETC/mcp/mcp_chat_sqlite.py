import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sqlite3
import re

# OpenAI 클라이언트 임포트 (새 버전)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI not available, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0"])
    from openai import OpenAI
    OPENAI_AVAILABLE = True

# MCP 클라이언트 임포트 (선택사항)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class SQLChatBot:
    """자연어를 SQL로 변환하고 실행하는 챗봇"""
    
    def __init__(self, db_path: str, openai_api_key: str = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # OpenAI API 키 설정
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # OpenAI 클라이언트 초기화 (새 API 버전)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # 데이터베이스 스키마 정보 저장
        self.schema_info = self._get_schema_info()
        
        print(f"💾 Database: {self.db_path}")
        print(f"🤖 OpenAI API: {'✅ Connected' if self.openai_api_key else '❌ Not available'}")
    
    def _get_schema_info(self) -> str:
        """데이터베이스 스키마 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 목록 조회
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                schema_info = "Database Schema:\n"
                
                for (table_name,) in tables:
                    schema_info += f"\nTable: {table_name}\n"
                    
                    # 테이블 스키마 조회
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        col_name, col_type, not_null, default, pk = col[1], col[2], col[3], col[4], col[5]
                        pk_str = " (PRIMARY KEY)" if pk else ""
                        not_null_str = " NOT NULL" if not_null else ""
                        default_str = f" DEFAULT {default}" if default else ""
                        schema_info += f"  - {col_name}: {col_type}{pk_str}{not_null_str}{default_str}\n"
                    
                    # 샘플 데이터 조회 (처음 3개 행)
                    try:
                        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                        sample_rows = cursor.fetchall()
                        
                        if sample_rows:
                            schema_info += f"  Sample data:\n"
                            for i, row in enumerate(sample_rows, 1):
                                schema_info += f"    {i}. {row}\n"
                    except:
                        pass
                
                return schema_info
                
        except Exception as e:
            return f"Error getting schema: {str(e)}"
    
    def _split_sql_statements(self, sql_text: str) -> List[str]:
        """여러 SQL 문을 개별 문으로 분리"""
        # 기본적인 정리
        sql_text = sql_text.strip()
        if not sql_text:
            return []
        
        # 주석 제거 (-- 스타일)
        lines = sql_text.split('\n')
        cleaned_lines = []
        for line in lines:
            if '--' in line:
                line = line[:line.find('--')]
            cleaned_lines.append(line)
        sql_text = '\n'.join(cleaned_lines)
        
        # /* */ 스타일 주석 제거
        sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
        
        # 간단한 세미콜론 분리 (문자열 내부 고려하지 않는 간단한 버전)
        statements = []
        current_statement = ""
        
        for char in sql_text:
            if char == ';':
                current_statement = current_statement.strip()
                if current_statement:
                    statements.append(current_statement)
                current_statement = ""
            else:
                current_statement += char
        
        # 마지막 문 추가 (세미콜론이 없는 경우)
        current_statement = current_statement.strip()
        if current_statement:
            statements.append(current_statement)
        
        return [stmt for stmt in statements if stmt.strip()]
    
    def execute_sql(self, sql: str, params: List = None) -> Dict[str, Any]:
        """SQL 실행 (다중 문 지원)"""
        try:
            sql = sql.strip()
            if not sql:
                return {"success": False, "error": "Empty SQL statement"}
            
            # SQL 문이 여러 개인지 확인
            statements = self._split_sql_statements(sql)
            
            if len(statements) > 1:
                return self._execute_multiple_statements(statements)
            else:
                return self._execute_single_statement(sql, params)
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_single_statement(self, sql: str, params: List = None) -> Dict[str, Any]:
        """단일 SQL 문 실행"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                sql_upper = sql.strip().upper()
                
                # SELECT 쿼리인지 확인
                if sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA"):
                    rows = cursor.fetchall()
                    results = [dict(row) for row in rows]
                    return {
                        "success": True, 
                        "results": results, 
                        "row_count": len(results),
                        "query_type": "SELECT"
                    }
                # DDL 쿼리들 (CREATE, DROP, ALTER 등)
                elif any(sql_upper.startswith(cmd) for cmd in ["CREATE", "DROP", "ALTER"]):
                    conn.commit()
                    return {
                        "success": True,
                        "query_type": "DDL",
                        "message": "DDL operation completed successfully"
                    }
                # DML 쿼리들 (INSERT, UPDATE, DELETE)
                else:
                    conn.commit()
                    return {
                        "success": True, 
                        "affected_rows": cursor.rowcount,
                        "query_type": "DML"
                    }
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_multiple_statements(self, statements: List[str]) -> Dict[str, Any]:
        """여러 SQL 문 실행"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                results = []
                total_affected_rows = 0
                last_select_results = None
                successful_count = 0
                
                for i, statement in enumerate(statements):
                    statement = statement.strip()
                    if not statement:
                        continue
                    
                    try:
                        cursor.execute(statement)
                        statement_upper = statement.upper()
                        
                        if statement_upper.startswith("SELECT") or statement_upper.startswith("PRAGMA"):
                            rows = cursor.fetchall()
                            last_select_results = [dict(row) for row in rows]
                            results.append({
                                "statement": statement,
                                "type": "SELECT",
                                "results": last_select_results,
                                "row_count": len(last_select_results),
                                "success": True
                            })
                            successful_count += 1
                        elif any(statement_upper.startswith(cmd) for cmd in ["CREATE", "DROP", "ALTER"]):
                            results.append({
                                "statement": statement,
                                "type": "DDL",
                                "message": "DDL operation completed successfully",
                                "success": True
                            })
                            successful_count += 1
                        else:  # DML
                            affected = cursor.rowcount
                            total_affected_rows += affected
                            results.append({
                                "statement": statement,
                                "type": "DML",
                                "affected_rows": affected,
                                "success": True
                            })
                            successful_count += 1
                    
                    except Exception as e:
                        results.append({
                            "statement": statement,
                            "type": "ERROR",
                            "error": str(e),
                            "success": False
                        })
                        # 에러가 발생하면 롤백하고 중단
                        conn.rollback()
                        break
                
                # 성공한 경우에만 커밋
                if successful_count > 0 and all(r.get("success", False) for r in results):
                    conn.commit()
                
                failed_count = len([r for r in results if not r.get("success", True)])
                
                return {
                    "success": True,
                    "query_type": "MULTIPLE",
                    "total_statements": len(statements),
                    "successful_statements": successful_count,
                    "failed_statements": failed_count,
                    "results": results,
                    "last_select_results": last_select_results,
                    "total_affected_rows": total_affected_rows
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def natural_language_to_sql(self, user_question: str) -> Dict[str, Any]:
        """자연어를 SQL로 변환"""
        try:
            system_prompt = f"""You are a SQL expert assistant. Convert natural language questions to SQL queries.

{self.schema_info}

Rules:
1. Generate ONLY valid SQLite SQL queries
2. Use proper SQLite syntax
3. Always use table and column names exactly as shown in the schema
4. For Korean text, understand the meaning and convert appropriately
5. Support DDL operations (CREATE, DROP, ALTER TABLE)
6. Support DML operations (INSERT, UPDATE, DELETE)
7. Support DQL operations (SELECT)
8. If the question is ambiguous, make reasonable assumptions
9. For aggregations, use proper GROUP BY clauses
10. You can return multiple SQL statements separated by semicolons if needed
11. Return only the SQL query/queries, no explanations unless asked

Examples:
- "모든 직원을 보여줘" → "SELECT * FROM employees;"
- "새 테이블을 만들어줘" → "CREATE TABLE new_table (id INTEGER PRIMARY KEY, name TEXT);"
- "여러 테이블을 만들어줘" → "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER);"
"""
            
            # 새 OpenAI API 사용
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # 코드 블록 제거 (```sql로 감싸진 경우)
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            
            return {
                "success": True,
                "sql": sql_query,
                "original_question": user_question
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_question": user_question
            }
    
    def format_results(self, results: List[Dict], max_rows: int = 10) -> str:
        """결과를 보기 좋게 포맷팅"""
        if not results:
            return "📋 No results found."
        
        output = f"📊 Results ({len(results)} rows"
        if len(results) > max_rows:
            output += f", showing first {max_rows}"
        output += "):\n\n"
        
        # 컬럼 헤더
        headers = list(results[0].keys())
        output += "| " + " | ".join(f"{h:>12}" for h in headers) + " |\n"
        output += "|" + "|".join("-" * 14 for _ in headers) + "|\n"
        
        # 데이터 행
        display_results = results[:max_rows]
        for row in display_results:
            formatted_row = []
            for header in headers:
                value = row[header]
                if value is None:
                    formatted_value = "NULL"
                elif isinstance(value, float):
                    formatted_value = f"{value:,.1f}"
                elif isinstance(value, int) and value > 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)
                
                # 길이 제한
                if len(formatted_value) > 12:
                    formatted_value = formatted_value[:9] + "..."
                
                formatted_row.append(f"{formatted_value:>12}")
            
            output += "| " + " | ".join(formatted_row) + " |\n"
        
        if len(results) > max_rows:
            output += f"\n... and {len(results) - max_rows} more rows\n"
        
        return output
    
    def format_multiple_results(self, exec_result: Dict[str, Any]) -> str:
        """다중 SQL 문 실행 결과 포맷팅"""
        results = exec_result["results"]
        output = f"📊 **다중 SQL 문 실행 결과**\n"
        output += f"총 {exec_result['total_statements']}개 문 중 {exec_result['successful_statements']}개 성공"
        if exec_result['failed_statements'] > 0:
            output += f", {exec_result['failed_statements']}개 실패"
        output += "\n\n"
        
        for i, result in enumerate(results, 1):
            status = "✅" if result.get("success", True) else "❌"
            output += f"**{i}. {status} {result['type']} 문:**\n"
            output += f"```sql\n{result['statement']}\n```\n"
            
            if result["type"] == "SELECT":
                if result.get("results"):
                    output += self.format_results(result["results"])
                else:
                    output += "📋 No results found.\n"
            elif result["type"] == "DDL":
                if result.get("success", True):
                    output += f"✅ {result.get('message', 'DDL 작업 완료')}\n"
                else:
                    output += f"❌ 오류: {result.get('error', 'Unknown error')}\n"
            elif result["type"] == "DML":
                if result.get("success", True):
                    output += f"✅ {result.get('affected_rows', 0)}개 행이 영향받았습니다.\n"
                else:
                    output += f"❌ 오류: {result.get('error', 'Unknown error')}\n"
            elif result["type"] == "ERROR":
                output += f"❌ 오류: {result.get('error', 'Unknown error')}\n"
            
            output += "\n"
        
        # 요약 정보
        if exec_result.get("last_select_results"):
            output += f"💡 **마지막 SELECT 결과**: {len(exec_result['last_select_results'])}개 행\n"
        
        if exec_result.get("total_affected_rows", 0) > 0:
            output += f"💡 **총 영향받은 행**: {exec_result['total_affected_rows']}개\n"
        
        return output
    
    def get_sql_explanation(self, sql: str) -> str:
        """SQL 쿼리 설명 생성"""
        try:
            # 다중 문인지 확인
            statements = self._split_sql_statements(sql)
            
            if len(statements) > 1:
                explanation_prompt = f"이 여러 개의 SQL 쿼리들이 무엇을 하는지 간단히 설명해주세요:\n{sql}"
            else:
                explanation_prompt = f"이 SQL 쿼리가 무엇을 하는지 간단히 설명해주세요:\n{sql}"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "Explain SQL queries in simple Korean. Be concise and focus on what the query does."
                    },
                    {
                        "role": "user", 
                        "content": explanation_prompt
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"설명을 생성할 수 없습니다: {str(e)}"
    
    async def chat(self, user_input: str) -> str:
        """챗봇 메인 대화 함수"""
        if not user_input.strip():
            return "❓ 질문을 입력해주세요."
        
        # 특수 명령어 처리
        if user_input.lower() in ['help', '도움말']:
            return self._get_help_message()
        
        if user_input.lower() in ['schema', '스키마']:
            return f"📋 Database Schema:\n\n{self.schema_info}"
        
        if user_input.lower() in ['tables', '테이블', '테이블목록']:
            return await self._show_tables()
        
        if user_input.lower().startswith('describe ') or user_input.lower().startswith('desc '):
            table_name = user_input.split()[1]
            return await self._describe_table(table_name)
        
        if user_input.lower().startswith('sql:'):
            # 직접 SQL 실행
            sql = user_input[4:].strip()
            return await self._execute_direct_sql(sql)
        
        # 위험한 작업 확인
        dangerous_keywords = ['drop table', 'delete from', 'truncate']
        if any(keyword in user_input.lower() for keyword in dangerous_keywords):
            return await self._handle_dangerous_operation(user_input)
        
        # 자연어 → SQL 변환
        print(f"🤔 Understanding: {user_input}")
        
        # 1. 자연어를 SQL로 변환
        sql_result = self.natural_language_to_sql(user_input)
        
        if not sql_result["success"]:
            return f"❌ SQL 변환 실패: {sql_result['error']}"
        
        sql_query = sql_result["sql"]
        
        # 2. SQL 설명 생성
        explanation = self.get_sql_explanation(sql_query)
        
        # 3. SQL 실행
        print(f"🔍 Generated SQL: {sql_query}")
        exec_result = self.execute_sql(sql_query)
        
        # 4. 결과 포맷팅
        response = f"💭 **질문**: {user_input}\n\n"
        response += f"🔍 **생성된 SQL**:\n```sql\n{sql_query}\n```\n\n"
        response += f"📖 **설명**: {explanation}\n\n"
        
        if exec_result["success"]:
            if exec_result.get("query_type") == "SELECT":
                results = exec_result["results"]
                response += self.format_results(results)
                
                # 요약 정보 추가
                if results:
                    response += f"\n💡 **요약**: {len(results)}개의 결과를 찾았습니다."
                else:
                    response += f"\n💡 **요약**: 조건에 맞는 데이터가 없습니다."
                    
            elif exec_result.get("query_type") == "DDL":
                response += f"✅ **DDL 실행 성공**: {exec_result.get('message', 'DDL 작업이 완료되었습니다.')}"
                # 스키마 정보 업데이트
                self.schema_info = self._get_schema_info()
                response += "\n📝 **스키마 정보가 업데이트되었습니다.**"
                
            elif exec_result.get("query_type") == "MULTIPLE":
                response += self.format_multiple_results(exec_result)
                # 스키마 정보 업데이트 (DDL이 포함된 경우)
                if any(r.get("type") == "DDL" for r in exec_result["results"]):
                    self.schema_info = self._get_schema_info()
                    response += "\n📝 **스키마 정보가 업데이트되었습니다.**"
                
            else:  # DML
                affected = exec_result.get("affected_rows", 0)
                response += f"✅ **DML 실행 성공**: {affected}개 행이 영향받았습니다."
        else:
            response += f"❌ **실행 실패**: {exec_result['error']}"
        
        return response
    
    async def _execute_direct_sql(self, sql: str) -> str:
        """직접 SQL 실행"""
        result = self.execute_sql(sql)
        
        response = f"🔍 **직접 SQL 실행**:\n```sql\n{sql}\n```\n\n"
        
        if result["success"]:
            if result.get("query_type") == "SELECT":
                results = result["results"]
                response += self.format_results(results)
            elif result.get("query_type") == "MULTIPLE":
                response += self.format_multiple_results(result)
            elif result.get("query_type") == "DDL":
                response += f"✅ **DDL 실행 성공**: {result.get('message', 'DDL 작업이 완료되었습니다.')}"
                # 스키마 정보 업데이트
                self.schema_info = self._get_schema_info()
                response += "\n📝 **스키마 정보가 업데이트되었습니다.**"
            else:
                affected = result.get("affected_rows", 0)
                response += f"✅ **실행 성공**: {affected}개 행이 영향받았습니다."
        else:
            response += f"❌ **실행 실패**: {result['error']}"
        
        return response
    
    async def _show_tables(self) -> str:
        """모든 테이블 목록 표시"""
        result = self.execute_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        
        if result["success"]:
            tables = result["results"]
            if tables:
                response = "📋 **데이터베이스 테이블 목록**:\n\n"
                for i, table in enumerate(tables, 1):
                    table_name = table['name']
                    # 각 테이블의 행 개수 조회
                    count_result = self.execute_sql(f"SELECT COUNT(*) as count FROM {table_name}")
                    row_count = count_result["results"][0]["count"] if count_result["success"] else "?"
                    response += f"{i}. **{table_name}** ({row_count} rows)\n"
                
                response += f"\n💡 총 {len(tables)}개의 테이블이 있습니다."
                response += "\n📖 특정 테이블 구조 보기: `describe 테이블명`"
                return response
            else:
                return "📋 데이터베이스에 테이블이 없습니다."
        else:
            return f"❌ 테이블 목록 조회 실패: {result['error']}"
    
    async def _describe_table(self, table_name: str) -> str:
        """특정 테이블의 구조 표시"""
        # 테이블 스키마 조회
        schema_result = self.execute_sql(f"PRAGMA table_info({table_name})")
        
        if not schema_result["success"]:
            return f"❌ 테이블 '{table_name}'을 찾을 수 없습니다."
        
        columns = schema_result["results"]
        if not columns:
            return f"❌ 테이블 '{table_name}'이 존재하지 않습니다."
        
        response = f"📋 **테이블 '{table_name}' 구조**:\n\n"
        response += "| Column | Type | Null | Default | PK |\n"
        response += "|--------|------|------|---------|----|\n"
        
        for col in columns:
            col_name = col['name']
            col_type = col['type']
            not_null = "NO" if col['notnull'] else "YES"
            default = col['dflt_value'] or "-"
            pk = "✓" if col['pk'] else ""
            
            response += f"| {col_name} | {col_type} | {not_null} | {default} | {pk} |\n"
        
        # 샘플 데이터 표시
        sample_result = self.execute_sql(f"SELECT * FROM {table_name} LIMIT 3")
        if sample_result["success"] and sample_result["results"]:
            response += f"\n📊 **샘플 데이터** (처음 3개 행):\n"
            response += self.format_results(sample_result["results"], max_rows=3)
        
        # 행 개수 표시
        count_result = self.execute_sql(f"SELECT COUNT(*) as total FROM {table_name}")
        if count_result["success"]:
            total_rows = count_result["results"][0]["total"]
            response += f"\n💡 총 **{total_rows}**개의 행이 있습니다."
        
        return response
    
    async def _handle_dangerous_operation(self, user_input: str) -> str:
        """위험한 작업 처리"""
        response = f"⚠️  **위험한 작업 감지**\n\n"
        response += f"요청: {user_input}\n\n"
        response += "이 작업은 데이터를 삭제하거나 구조를 변경할 수 있습니다.\n"
        response += "정말 실행하시겠습니까?\n\n"
        response += "계속하려면 다음과 같이 입력하세요:\n"
        response += f"`sql: {user_input}`\n\n"
        response += "💡 **팁**: 중요한 데이터는 백업 후 작업하세요!"
        
        return response
    
    def _get_help_message(self) -> str:
        """도움말 메시지"""
        return """🤖 **SQL 챗봇 도움말**

**자연어 질문 예시:**
• "모든 직원을 보여줘"
• "개발팀에서 일하는 사람들은?"
• "연봉이 가장 높은 직원 3명은?"
• "부서별 평균 연봉은?"
• "2023년에 입사한 직원들은?"
• "김철수의 정보를 보여줘"

**다중 SQL 문 지원:**
• "여러 테이블을 만들어줘"
• "데이터를 삽입하고 조회해줘"
• 세미콜론으로 구분된 여러 SQL 문을 한 번에 실행 가능

**특수 명령어:**
• `help` 또는 `도움말` - 이 도움말 표시
• `schema` 또는 `스키마` - 데이터베이스 구조 보기
• `tables` 또는 `테이블목록` - 모든 테이블 목록 보기
• `describe 테이블명` - 특정 테이블 구조 보기
• `sql: SELECT * FROM employees` - 직접 SQL 실행

**팁:**
• 한국어와 영어 모두 지원합니다
• 복잡한 질문도 가능합니다
• 여러 SQL 문을 한 번에 실행할 수 있습니다
• SQL을 모르셔도 자연스럽게 질문하세요!
"""

async def setup_sample_database(db_path: str):
    """샘플 데이터베이스 설정"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    department TEXT NOT NULL,
                    salary REAL,
                    hire_date DATE,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 샘플 데이터 확인
            cursor.execute("SELECT COUNT(*) FROM employees")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # 샘플 데이터 삽입
                sample_employees = [
                    ("김철수", "kim.cs@company.com", "개발팀", 75000000, "2023-01-15"),
                    ("이영희", "lee.yh@company.com", "마케팅팀", 65000000, "2023-02-20"),
                    ("박민수", "park.ms@company.com", "인사팀", 70000000, "2023-03-10"),
                    ("정수진", "jung.sj@company.com", "개발팀", 80000000, "2022-11-05"),
                    ("최동현", "choi.dh@company.com", "영업팀", 72000000, "2023-04-12"),
                    ("한지영", "han.jy@company.com", "재무팀", 78000000, "2022-12-20"),
                    ("강민호", "kang.mh@company.com", "개발팀", 68000000, "2023-05-08"),
                    ("윤서연", "yoon.sy@company.com", "마케팅팀", 63000000, "2023-06-15"),
                    ("임현우", "lim.hw@company.com", "개발팀", 77000000, "2022-09-20"),
                    ("송지은", "song.je@company.com", "재무팀", 74000000, "2023-07-10")
                ]
                
                cursor.executemany("""
                    INSERT INTO employees (name, email, department, salary, hire_date)
                    VALUES (?, ?, ?, ?, ?)
                """, sample_employees)
                
                conn.commit()
                print(f"✅ 샘플 데이터 {len(sample_employees)}개 생성 완료")
            else:
                print(f"✅ 기존 데이터 {count}개 발견")
                
    except Exception as e:
        print(f"❌ 데이터베이스 설정 실패: {e}")

def test_openai_connection(api_key: str) -> bool:
    """OpenAI API 연결 테스트"""
    try:
        client = OpenAI(api_key=api_key)
        
        # 간단한 테스트 요청
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API 연결 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API 연결 테스트 실패: {e}")
        return False

async def main():
    """메인 함수"""
    print("🤖 자연어 SQL 변환 챗봇 (다중 SQL 문 지원)")
    print("="*60)
    
    # OpenAI API 키 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("💡 사용법:")
        print("   Windows: set OPENAI_API_KEY=your_api_key_here")
        print("   Linux/Mac: export OPENAI_API_KEY=your_api_key_here")
        print("   또는 .env 파일을 사용하세요")
        
        # 직접 입력 받기
        openai_key = input("\nOpenAI API 키를 직접 입력하세요: ").strip()
        if not openai_key:
            print("❌ API 키가 필요합니다. 프로그램을 종료합니다.")
            return
    
    # OpenAI API 연결 테스트
    print("🔗 OpenAI API 연결 테스트 중...")
    if not test_openai_connection(openai_key):
        print("❌ OpenAI API 연결에 실패했습니다. API 키를 확인해주세요.")
        return
    
    # 데이터베이스 경로 설정
    db_path = Path.home() / "Documents" / "sqlite" / "employees.db"
    
    try:
        # 샘플 데이터베이스 설정
        print("📋 데이터베이스 설정 중...")
        await setup_sample_database(str(db_path))
        
        # 챗봇 초기화
        chatbot = SQLChatBot(str(db_path), openai_key)
        
        print("\n🚀 챗봇이 준비되었습니다!")
        print("💬 자연어로 질문하거나 'help'를 입력하세요")
        print("🔧 여러 SQL 문도 한 번에 실행 가능합니다")
        print("🚪 종료하려면 'quit' 또는 'exit'를 입력하세요")
        print("-" * 60)
        
        # 샘플 질문 제안
        print("\n💡 시작하기 좋은 질문들:")
        sample_questions = [
            "모든 직원을 보여줘",
            "개발팀 직원들만 보여줘", 
            "연봉이 가장 높은 직원 3명은?",
            "부서별 평균 연봉은?",
            "2023년에 입사한 직원들은?",
            "새 테이블을 만들고 데이터를 넣어줘",
            "여러 테이블을 한 번에 만들어줘"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"   {i}. {question}")
        
        # 대화 루프
        while True:
            try:
                user_input = input("\n🗣️  사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("👋 챗봇을 종료합니다. 좋은 하루 되세요!")
                    break
                
                if not user_input:
                    continue
                
                # 빠른 선택 (숫자 입력)
                if user_input.isdigit():
                    num = int(user_input)
                    if 1 <= num <= len(sample_questions):
                        user_input = sample_questions[num - 1]
                        print(f"선택된 질문: {user_input}")
                    else:
                        print("❓ 잘못된 번호입니다. 다시 입력해주세요.")
                        continue
                
                # 챗봇 응답 생성
                print("\n🤖 AI가 생각 중...")
                response = await chatbot.chat(user_input)
                print(f"\n🤖 챗봇:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("💡 다시 시도해보세요.")
    
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")

# 환경변수 로드를 위한 선택적 python-dotenv 지원
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if __name__ == "__main__":
    asyncio.run(main())