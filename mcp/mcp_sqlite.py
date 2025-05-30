import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# MCP 클라이언트 임포트
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("MCP not available, will use alternative methods")
    MCP_AVAILABLE = False

# SQLite 직접 사용을 위한 import
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

class SQLiteMCPManager:
    """SQLite MCP 서버 관리 클래스"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # MCP 서버 설정
        self.server_config = {
            "command": "npx",
            "args": ["-y", "mcp-server-sqlite-npx", str(self.db_path)]
        }
        
        print(f"Database path: {self.db_path}")
    
    async def check_mcp_availability(self) -> bool:
        """MCP 서버가 사용 가능한지 확인"""
        try:
            result = subprocess.run(["npx", "--version"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def execute_sql_with_mcp(self, sql: str, params: List = None) -> Dict[str, Any]:
        """MCP를 사용하여 SQL 실행"""
        try:
            server_params = StdioServerParameters(
                command=self.server_config["command"],
                args=self.server_config["args"],
                env={}
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # 사용 가능한 도구 확인
                    tools = await session.list_tools()
                    print(f"Available tools: {[tool.name for tool in tools.tools]}")
                    
                    # SQL 실행 도구 찾기
                    sql_tool = None
                    for tool in tools.tools:
                        if any(keyword in tool.name.lower() for keyword in ["sql", "query", "execute"]):
                            sql_tool = tool
                            break
                    
                    if not sql_tool:
                        return {"success": False, "error": "No SQL execution tool found"}
                    
                    # SQL 실행
                    tool_args = {"query": sql}
                    if params:
                        tool_args["parameters"] = params
                    
                    result = await session.call_tool(sql_tool.name, tool_args)
                    
                    return {
                        "success": True,
                        "result": result.content,
                        "tool_used": sql_tool.name
                    }
                    
        except Exception as e:
            print(f"MCP SQL execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_sql_direct(self, sql: str, params: List = None) -> Dict[str, Any]:
        """직접 SQLite를 사용하여 SQL 실행"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                # SELECT 쿼리인지 확인
                if sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    results = [dict(row) for row in rows]
                    return {"success": True, "results": results, "row_count": len(results)}
                else:
                    conn.commit()
                    return {"success": True, "affected_rows": cursor.rowcount}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_sql(self, sql: str, params: List = None, use_mcp: bool = True) -> Dict[str, Any]:
        """SQL 실행 (MCP 우선, 실패시 직접 실행)"""
        if use_mcp and MCP_AVAILABLE:
            print(f"🔧 Executing SQL with MCP: {sql[:50]}...")
            result = await self.execute_sql_with_mcp(sql, params)
            
            if result["success"]:
                return result
            else:
                print(f"MCP failed: {result['error']}, falling back to direct SQLite")
        
        if SQLITE_AVAILABLE:
            print(f"🔧 Executing SQL directly: {sql[:50]}...")
            return self.execute_sql_direct(sql, params)
        else:
            return {"success": False, "error": "Neither MCP nor direct SQLite available"}

async def create_employee_table(db_manager: SQLiteMCPManager) -> bool:
    """직원 테이블 생성"""
    create_table_sql = """
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
    """
    
    print("📋 Creating employee table...")
    result = await db_manager.execute_sql(create_table_sql)
    
    if result["success"]:
        print("✅ Employee table created successfully")
        return True
    else:
        print(f"❌ Failed to create table: {result['error']}")
        return False

async def insert_sample_employees(db_manager: SQLiteMCPManager) -> bool:
    """샘플 직원 데이터 삽입"""
    sample_employees = [
        ("김철수", "kim.cs@company.com", "개발팀", 75000000, "2023-01-15"),
        ("이영희", "lee.yh@company.com", "마케팅팀", 65000000, "2023-02-20"),
        ("박민수", "park.ms@company.com", "인사팀", 70000000, "2023-03-10"),
        ("정수진", "jung.sj@company.com", "개발팀", 80000000, "2022-11-05"),
        ("최동현", "choi.dh@company.com", "영업팀", 72000000, "2023-04-12"),
        ("한지영", "han.jy@company.com", "재무팀", 78000000, "2022-12-20"),
        ("강민호", "kang.mh@company.com", "개발팀", 68000000, "2023-05-08"),
        ("윤서연", "yoon.sy@company.com", "마케팅팀", 63000000, "2023-06-15")
    ]
    
    insert_sql = """
    INSERT INTO employees (name, email, department, salary, hire_date)
    VALUES (?, ?, ?, ?, ?)
    """
    
    print("👥 Inserting sample employee data...")
    success_count = 0
    
    for employee in sample_employees:
        result = await db_manager.execute_sql(insert_sql, list(employee))
        
        if result["success"]:
            success_count += 1
            print(f"   ✅ Inserted: {employee[0]} ({employee[2]})")
        else:
            print(f"   ❌ Failed to insert {employee[0]}: {result['error']}")
    
    print(f"📊 Successfully inserted {success_count}/{len(sample_employees)} employees")
    return success_count > 0

async def query_employees(db_manager: SQLiteMCPManager):
    """직원 데이터 조회"""
    queries = [
        {
            "name": "All Employees",
            "sql": "SELECT * FROM employees ORDER BY hire_date DESC",
            "description": "모든 직원 조회 (입사일 내림차순)"
        },
        {
            "name": "Developers Only",
            "sql": "SELECT name, email, salary FROM employees WHERE department = '개발팀' ORDER BY salary DESC",
            "description": "개발팀 직원만 조회 (연봉 내림차순)"
        },
        {
            "name": "High Salary Employees", 
            "sql": "SELECT name, department, salary FROM employees WHERE salary > 70000000 ORDER BY salary DESC",
            "description": "연봉 7천만원 이상 직원 조회"
        },
        {
            "name": "Department Summary",
            "sql": """
            SELECT 
                department,
                COUNT(*) as employee_count,
                AVG(salary) as avg_salary,
                MAX(salary) as max_salary,
                MIN(salary) as min_salary
            FROM employees 
            GROUP BY department 
            ORDER BY avg_salary DESC
            """,
            "description": "부서별 통계"
        },
        {
            "name": "Recent Hires",
            "sql": "SELECT name, department, hire_date FROM employees WHERE hire_date >= '2023-01-01' ORDER BY hire_date DESC",
            "description": "2023년 이후 입사자"
        }
    ]
    
    print("\n" + "="*70)
    print("📊 EMPLOYEE DATA QUERIES")
    print("="*70)
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n{i}. {query_info['name']}")
        print(f"   📝 {query_info['description']}")
        print(f"   🔍 SQL: {' '.join(query_info['sql'].split())}")
        
        result = await db_manager.execute_sql(query_info["sql"])
        
        if result["success"]:
            if "results" in result:
                rows = result["results"]
                if rows:
                    print(f"   📋 Results ({len(rows)} rows):")
                    
                    # 결과가 너무 많으면 처음 5개만 표시
                    display_rows = rows[:5] if len(rows) > 5 else rows
                    
                    for j, row in enumerate(display_rows):
                        print(f"      {j+1}. {dict(row)}")
                    
                    if len(rows) > 5:
                        print(f"      ... and {len(rows) - 5} more rows")
                else:
                    print("   📋 No results found")
            else:
                print(f"   ✅ Query executed, affected rows: {result.get('affected_rows', 'N/A')}")
        else:
            print(f"   ❌ Query failed: {result['error']}")
        
        print("   " + "-"*60)

async def interactive_sql_mode(db_manager: SQLiteMCPManager):
    """대화형 SQL 모드"""
    print("\n" + "="*70)
    print("🔧 INTERACTIVE SQL MODE")
    print("="*70)
    print("Enter SQL commands (type 'exit' to quit, 'help' for examples)")
    print("-"*70)
    
    example_queries = [
        "SELECT COUNT(*) FROM employees",
        "SELECT department, COUNT(*) FROM employees GROUP BY department",
        "UPDATE employees SET salary = salary * 1.1 WHERE department = '개발팀'",
        "INSERT INTO employees (name, email, department, salary, hire_date) VALUES ('홍길동', 'hong@company.com', 'IT팀', 75000000, '2024-01-01')",
        "DELETE FROM employees WHERE id = 1"
    ]
    
    while True:
        try:
            user_input = input("\nSQL> ").strip()
            
            if user_input.lower() in ['exit', 'quit', '종료']:
                print("👋 Exiting interactive mode...")
                break
            
            elif user_input.lower() == 'help':
                print("\n📚 Example queries:")
                for i, example in enumerate(example_queries, 1):
                    print(f"  {i}. {example}")
                continue
            
            elif not user_input:
                continue
            
            # SQL 실행
            result = await db_manager.execute_sql(user_input)
            
            if result["success"]:
                if "results" in result:
                    rows = result["results"]
                    print(f"✅ Query successful! ({len(rows)} rows)")
                    for i, row in enumerate(rows[:10]):  # 최대 10개만 표시
                        print(f"  {i+1}. {dict(row)}")
                    if len(rows) > 10:
                        print(f"  ... and {len(rows) - 10} more rows")
                else:
                    print(f"✅ Query executed! Affected rows: {result.get('affected_rows', 'N/A')}")
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

async def main():
    """메인 함수"""
    print("🗄️  SQLite MCP Employee Management System")
    print("="*70)
    
    # 데이터베이스 파일 경로 설정
    db_path = Path.home() / "Documents" / "sqlite" / "employees.db"
    db_manager = SQLiteMCPManager(str(db_path))
    
    # MCP 사용 가능 여부 확인
    if MCP_AVAILABLE:
        mcp_available = await db_manager.check_mcp_availability()
        print(f"✅ MCP Available: {mcp_available}")
    else:
        print("⚠️  MCP not available, using direct SQLite")
    
    print(f"✅ SQLite Available: {SQLITE_AVAILABLE}")
    print()
    
    try:
        # 1. 테이블 생성
        table_created = await create_employee_table(db_manager)
        if not table_created:
            print("❌ Cannot proceed without table creation")
            return
        
        # 2. 샘플 데이터 삽입 (선택)
        print("\n" + "="*50)
        insert_choice = input("Insert sample employee data? (y/N): ").strip().lower()
        
        if insert_choice in ['y', 'yes']:
            await insert_sample_employees(db_manager)
        
        # 3. 데이터 조회
        await query_employees(db_manager)
        
        # 4. 대화형 SQL 모드 (선택)
        print("\n" + "="*50)
        interactive_choice = input("Enter interactive SQL mode? (y/N): ").strip().lower()
        
        if interactive_choice in ['y', 'yes']:
            await interactive_sql_mode(db_manager)
        
        print(f"\n🎉 Database operations completed!")
        print(f"📁 Database file: {db_path}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())