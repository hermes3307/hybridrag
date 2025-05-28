#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGNEX1 데이터 시각화 및 분석 도구
수집된 데이터를 조회하고 통계를 시각화합니다.
"""

import os
import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
from collections import Counter
import warnings

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class LIGNEX1DataViewer:
    """LIGNEX1 데이터 뷰어 클래스"""
    
    def __init__(self, data_dir="lignex1_data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "lignex1_articles.db")
        
        if not os.path.exists(self.db_path):
            print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {self.db_path}")
            print("먼저 01.extract.py를 실행하여 데이터를 수집해주세요.")
            return
        
        print(f"📊 데이터베이스 연결: {self.db_path}")
    
    def get_basic_stats(self):
        """기본 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 기사 수
                cursor.execute('SELECT COUNT(*) FROM articles')
                total_count = cursor.fetchone()[0]
                
                # API 제공자별 통계
                cursor.execute('''
                    SELECT api_provider, COUNT(*) 
                    FROM articles 
                    WHERE api_provider IS NOT NULL
                    GROUP BY api_provider
                ''')
                provider_stats = dict(cursor.fetchall())
                
                # API 타입별 통계
                cursor.execute('''
                    SELECT api_type, COUNT(*) 
                    FROM articles 
                    WHERE api_type IS NOT NULL
                    GROUP BY api_type
                ''')
                api_type_stats = dict(cursor.fetchall())
                
                # 키워드별 통계
                cursor.execute('''
                    SELECT search_keyword, COUNT(*) 
                    FROM articles 
                    WHERE search_keyword IS NOT NULL
                    GROUP BY search_keyword
                    ORDER BY COUNT(*) DESC
                ''')
                keyword_stats = dict(cursor.fetchall())
                
                # 날짜별 수집 통계 (최근 30일)
                thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute('''
                    SELECT DATE(collected_at) as date, COUNT(*) 
                    FROM articles 
                    WHERE collected_at > ? AND collected_at IS NOT NULL
                    GROUP BY DATE(collected_at)
                    ORDER BY date
                ''', (thirty_days_ago,))
                daily_stats = dict(cursor.fetchall())
                
                # 최신 기사 5건
                cursor.execute('''
                    SELECT title, source, api_provider, collected_at
                    FROM articles 
                    WHERE title IS NOT NULL
                    ORDER BY created_at DESC 
                    LIMIT 5
                ''')
                recent_articles = cursor.fetchall()
                
                return {
                    'total_count': total_count,
                    'provider_stats': provider_stats,
                    'api_type_stats': api_type_stats,
                    'keyword_stats': keyword_stats,
                    'daily_stats': daily_stats,
                    'recent_articles': recent_articles
                }
                
        except Exception as e:
            print(f"❌ 통계 조회 중 오류: {e}")
            return None
    
    def print_basic_stats(self):
        """기본 통계 출력"""
        stats = self.get_basic_stats()
        if not stats:
            return
        
        print("\n" + "="*60)
        print("🎯 LIGNEX1 데이터 수집 현황")
        print("="*60)
        
        print(f"📈 총 수집된 기사: {stats['total_count']:,}개")
        
        # API 제공자별 통계
        if stats['provider_stats']:
            print("\n📊 API 제공자별 통계:")
            for provider, count in stats['provider_stats'].items():
                percentage = (count / stats['total_count']) * 100
                print(f"  • {provider}: {count:,}개 ({percentage:.1f}%)")
        
        # API 타입별 통계
        if stats['api_type_stats']:
            print("\n🔍 API 타입별 통계:")
            for api_type, count in stats['api_type_stats'].items():
                percentage = (count / stats['total_count']) * 100
                print(f"  • {api_type}: {count:,}개 ({percentage:.1f}%)")
        
        # 키워드별 통계 (상위 5개)
        if stats['keyword_stats']:
            print("\n🏷️ 검색 키워드별 통계 (상위 5개):")
            for i, (keyword, count) in enumerate(list(stats['keyword_stats'].items())[:5], 1):
                percentage = (count / stats['total_count']) * 100
                print(f"  {i}. {keyword}: {count:,}개 ({percentage:.1f}%)")
        
        # 일일 수집 현황 (최근 7일)
        if stats['daily_stats']:
            print("\n📅 최근 수집 현황:")
            recent_days = list(stats['daily_stats'].items())[-7:]
            for date, count in recent_days:
                print(f"  • {date}: {count:,}개")
        
        # 최신 기사
        if stats['recent_articles']:
            print("\n📰 최신 수집 기사 (5건):")
            for i, (title, source, provider, collected_at) in enumerate(stats['recent_articles'], 1):
                # 제목 길이 제한
                display_title = title[:50] + "..." if len(title) > 50 else title
                collected_date = collected_at[:10] if collected_at else "Unknown"
                print(f"  {i}. [{provider}] {display_title}")
                print(f"     출처: {source} | 수집일: {collected_date}")
    
    def show_detailed_articles(self, limit=20, keyword=None, provider=None):
        """상세 기사 목록 출력"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT title, description, link, source, api_provider, 
                           search_keyword, collected_at
                    FROM articles 
                    WHERE title IS NOT NULL
                '''
                params = []
                
                if keyword:
                    query += ' AND search_keyword LIKE ?'
                    params.append(f'%{keyword}%')
                
                if provider:
                    query += ' AND api_provider = ?'
                    params.append(provider)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor = conn.execute(query, params)
                articles = cursor.fetchall()
                
                if not articles:
                    print("📄 조건에 맞는 기사가 없습니다.")
                    return
                
                print(f"\n📰 기사 목록 ({len(articles)}건)")
                print("="*80)
                
                for i, (title, desc, link, source, provider, search_kw, collected_at) in enumerate(articles, 1):
                    print(f"\n[{i}] {title}")
                    print(f"🏷️  키워드: {search_kw} | 제공: {provider} | 출처: {source}")
                    
                    if desc:
                        display_desc = desc[:100] + "..." if len(desc) > 100 else desc
                        print(f"📝  {display_desc}")
                    
                    print(f"🔗  {link}")
                    
                    if collected_at:
                        print(f"📅  수집일: {collected_at[:19]}")
                    
                    if i < len(articles):
                        print("-" * 80)
                        
        except Exception as e:
            print(f"❌ 기사 조회 중 오류: {e}")
    
    def create_visualizations(self, save_plots=False):
        """데이터 시각화 생성"""
        stats = self.get_basic_stats()
        if not stats:
            return
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LIGNEX1 데이터 수집 현황 분석', fontsize=16, fontweight='bold')
        
        # 1. API 제공자별 파이 차트
        if stats['provider_stats']:
            providers = list(stats['provider_stats'].keys())
            counts = list(stats['provider_stats'].values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            axes[0, 0].pie(counts, labels=providers, autopct='%1.1f%%', colors=colors[:len(providers)])
            axes[0, 0].set_title('API 제공자별 분포', fontweight='bold')
        
        # 2. 키워드별 막대 그래프 (상위 8개)
        if stats['keyword_stats']:
            keywords = list(stats['keyword_stats'].keys())[:8]
            keyword_counts = list(stats['keyword_stats'].values())[:8]
            
            bars = axes[0, 1].bar(range(len(keywords)), keyword_counts, color='skyblue')
            axes[0, 1].set_title('검색 키워드별 수집량 (상위 8개)', fontweight='bold')
            axes[0, 1].set_xticks(range(len(keywords)))
            axes[0, 1].set_xticklabels(keywords, rotation=45, ha='right')
            axes[0, 1].set_ylabel('기사 수')
            
            # 막대 위에 숫자 표시
            for bar, count in zip(bars, keyword_counts):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(keyword_counts)*0.01,
                               f'{int(count)}', ha='center', va='bottom')
        
        # 3. 일별 수집 추이 (최근 14일)
        if stats['daily_stats']:
            dates = list(stats['daily_stats'].keys())[-14:]
            daily_counts = list(stats['daily_stats'].values())[-14:]
            
            axes[1, 0].plot(dates, daily_counts, marker='o', linewidth=2, markersize=6, color='#FF6B6B')
            axes[1, 0].set_title('일별 수집 추이 (최근 14일)', fontweight='bold')
            axes[1, 0].set_xlabel('날짜')
            axes[1, 0].set_ylabel('수집 기사 수')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. API 타입별 막대 그래프
        if stats['api_type_stats']:
            api_types = list(stats['api_type_stats'].keys())
            type_counts = list(stats['api_type_stats'].values())
            
            bars = axes[1, 1].bar(api_types, type_counts, color='lightgreen')
            axes[1, 1].set_title('API 타입별 분포', fontweight='bold')
            axes[1, 1].set_xlabel('API 타입')
            axes[1, 1].set_ylabel('기사 수')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 막대 위에 숫자 표시
            for bar, count in zip(bars, type_counts):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(type_counts)*0.01,
                               f'{int(count)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_dir = os.path.join(self.data_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_path = os.path.join(plot_dir, f"lignex1_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 그래프가 저장되었습니다: {plot_path}")
        
        plt.show()
    
    def export_to_excel(self, filename=None):
        """Excel 파일로 데이터 내보내기"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 모든 데이터 조회
                df = pd.read_sql_query('''
                    SELECT title, description, link, source, api_provider, 
                           api_type, search_keyword, collected_at, created_at
                    FROM articles 
                    ORDER BY created_at DESC
                ''', conn)
                
                if df.empty:
                    print("📄 내보낼 데이터가 없습니다.")
                    return
                
                # 파일명 생성
                if not filename:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"lignex1_data_{timestamp}.xlsx"
                
                export_path = os.path.join(self.data_dir, filename)
                
                # Excel 파일 생성
                with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                    # 전체 데이터
                    df.to_excel(writer, sheet_name='전체데이터', index=False)
                    
                    # 통계 시트
                    stats = self.get_basic_stats()
                    if stats:
                        # 제공자별 통계
                        provider_df = pd.DataFrame(list(stats['provider_stats'].items()), 
                                                 columns=['API제공자', '기사수'])
                        provider_df.to_excel(writer, sheet_name='제공자별통계', index=False)
                        
                        # 키워드별 통계
                        keyword_df = pd.DataFrame(list(stats['keyword_stats'].items()), 
                                                columns=['검색키워드', '기사수'])
                        keyword_df.to_excel(writer, sheet_name='키워드별통계', index=False)
                
                print(f"📊 Excel 파일이 생성되었습니다: {export_path}")
                print(f"📈 총 {len(df)}개의 기사가 포함되었습니다.")
                
        except Exception as e:
            print(f"❌ Excel 내보내기 중 오류: {e}")
    
    def search_articles(self, search_term, limit=10):
        """기사 검색"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT title, description, link, source, api_provider, collected_at
                    FROM articles 
                    WHERE title LIKE ? OR description LIKE ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (f'%{search_term}%', f'%{search_term}%', limit))
                
                results = cursor.fetchall()
                
                if not results:
                    print(f"🔍 '{search_term}' 검색 결과가 없습니다.")
                    return
                
                print(f"\n🔍 '{search_term}' 검색 결과 ({len(results)}건)")
                print("="*80)
                
                for i, (title, desc, link, source, provider, collected_at) in enumerate(results, 1):
                    print(f"\n[{i}] {title}")
                    print(f"🏷️  제공: {provider} | 출처: {source}")
                    
                    if desc:
                        display_desc = desc[:150] + "..." if len(desc) > 150 else desc
                        print(f"📝  {display_desc}")
                    
                    print(f"🔗  {link}")
                    
                    if collected_at:
                        print(f"📅  수집일: {collected_at[:19]}")
                    
                    if i < len(results):
                        print("-" * 80)
                        
        except Exception as e:
            print(f"❌ 검색 중 오류: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='LIGNEX1 데이터 뷰어')
    parser.add_argument('--data-dir', default='lignex1_data', help='데이터 디렉토리 경로')
    parser.add_argument('--stats', action='store_true', help='기본 통계 출력')
    parser.add_argument('--articles', type=int, default=0, help='기사 목록 출력 (개수 지정)')
    parser.add_argument('--keyword', type=str, help='특정 키워드로 필터링')
    parser.add_argument('--provider', type=str, choices=['naver', 'kakao'], help='API 제공자로 필터링')
    parser.add_argument('--plot', action='store_true', help='그래프 시각화')
    parser.add_argument('--save-plot', action='store_true', help='그래프를 파일로 저장')
    parser.add_argument('--excel', action='store_true', help='Excel 파일로 내보내기')
    parser.add_argument('--search', type=str, help='기사 검색')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    
    args = parser.parse_args()
    
    # 데이터 뷰어 초기화
    viewer = LIGNEX1DataViewer(args.data_dir)
    
    # 대화형 모드
    if args.interactive:
        interactive_mode(viewer)
        return
    
    # 기본 통계 출력
    if args.stats or (not any([args.articles, args.plot, args.excel, args.search])):
        viewer.print_basic_stats()
    
    # 기사 목록 출력
    if args.articles > 0:
        viewer.show_detailed_articles(
            limit=args.articles, 
            keyword=args.keyword, 
            provider=args.provider
        )
    
    # 그래프 시각화
    if args.plot or args.save_plot:
        try:
            viewer.create_visualizations(save_plots=args.save_plot)
        except ImportError:
            print("❌ matplotlib가 설치되지 않았습니다. 다음 명령어로 설치해주세요:")
            print("pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ 시각화 생성 중 오류: {e}")
    
    # Excel 내보내기
    if args.excel:
        try:
            viewer.export_to_excel()
        except ImportError:
            print("❌ pandas 또는 openpyxl이 설치되지 않았습니다. 다음 명령어로 설치해주세요:")
            print("pip install pandas openpyxl")
        except Exception as e:
            print(f"❌ Excel 내보내기 중 오류: {e}")
    
    # 기사 검색
    if args.search:
        viewer.search_articles(args.search)

def interactive_mode(viewer):
    """대화형 모드"""
    print("\n🎯 LIGNEX1 데이터 뷰어 - 대화형 모드")
    print("="*50)
    
    while True:
        print("\n📋 메뉴:")
        print("1. 기본 통계 출력")
        print("2. 기사 목록 보기")
        print("3. 기사 검색")
        print("4. 그래프 시각화")
        print("5. Excel 내보내기")
        print("0. 종료")
        
        try:
            choice = input("\n선택하세요 (0-5): ").strip()
            
            if choice == '0':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                viewer.print_basic_stats()
            elif choice == '2':
                limit = input("출력할 기사 수 (기본값: 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                viewer.show_detailed_articles(limit=limit)
            elif choice == '3':
                search_term = input("검색어를 입력하세요: ").strip()
                if search_term:
                    viewer.search_articles(search_term)
            elif choice == '4':
                try:
                    save = input("그래프를 파일로 저장하시겠습니까? (y/N): ").strip().lower()
                    viewer.create_visualizations(save_plots=(save == 'y'))
                except ImportError:
                    print("❌ matplotlib가 설치되지 않아 시각화를 사용할 수 없습니다.")
            elif choice == '5':
                try:
                    viewer.export_to_excel()
                except ImportError:
                    print("❌ pandas가 설치되지 않아 Excel 내보내기를 사용할 수 없습니다.")
            else:
                print("❌ 잘못된 선택입니다.")
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류: {e}")