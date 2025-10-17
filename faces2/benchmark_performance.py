#!/usr/bin/env python3
"""
성능 벤치마크 스크립트
이미지 다운로드, 특징 추출, 벡터 임베딩, DB 저장, 유사도 검색의 평균 처리 시간 측정
"""

import time
import os
from pathlib import Path
from core_backend import (
    IntegratedFaceSystem, FaceAnalyzer, FaceEmbedder,
    DatabaseManager, SystemConfig
)

def benchmark_performance():
    """각 단계별 성능 측정"""

    print("=" * 70)
    print("  성능 벤치마크 - Face Processing System")
    print("=" * 70)
    print()

    # 시스템 초기화
    print("시스템 초기화 중...")
    system = IntegratedFaceSystem()
    if not system.initialize():
        print("❌ 시스템 초기화 실패")
        return

    analyzer = FaceAnalyzer()
    embedder = FaceEmbedder()

    # 기존 이미지 파일 찾기
    face_files = list(Path(system.config.faces_dir).rglob("*.jpg"))[:10]  # 처음 10개만 테스트

    if not face_files:
        print("❌ 테스트할 이미지가 없습니다.")
        return

    print(f"✅ 테스트 이미지 {len(face_files)}개 발견\n")

    # 1. 이미지 다운로드 성능 측정
    print("1️⃣  이미지 다운로드 성능 측정")
    print("-" * 70)
    download_times = []

    print("   다운로드 3회 테스트 중...")
    for i in range(3):
        start = time.time()
        file_path = system.downloader.download_face()
        elapsed = time.time() - start

        if file_path:
            download_times.append(elapsed)
            print(f"   시도 {i+1}: {elapsed:.3f}초 - 성공")
        else:
            print(f"   시도 {i+1}: {elapsed:.3f}초 - 중복 또는 실패")

        time.sleep(1)  # 서버 부하 방지

    if download_times:
        avg_download = sum(download_times) / len(download_times)
        print(f"\n   📊 평균 다운로드 시간: {avg_download:.3f}초")
        print(f"   📊 처리량: {1/avg_download:.2f} 이미지/초\n")
    else:
        print(f"   ⚠️  새로운 다운로드 없음 (모두 중복)\n")

    # 2. 특징 추출 성능 측정
    print("2️⃣  특징 추출 (Feature Extraction) 성능 측정")
    print("-" * 70)
    analysis_times = []

    for i, file_path in enumerate(face_files[:5]):
        start = time.time()
        features = analyzer.analyze_face(str(file_path))
        elapsed = time.time() - start
        analysis_times.append(elapsed)
        print(f"   이미지 {i+1}: {elapsed:.4f}초")

    avg_analysis = sum(analysis_times) / len(analysis_times)
    print(f"\n   📊 평균 특징 추출 시간: {avg_analysis:.4f}초")
    print(f"   📊 처리량: {1/avg_analysis:.2f} 이미지/초\n")

    # 3. 벡터 임베딩 성능 측정
    print("3️⃣  벡터 임베딩 (Embedding) 성능 측정")
    print("-" * 70)
    embedding_times = []

    for i, file_path in enumerate(face_files[:5]):
        features = analyzer.analyze_face(str(file_path))

        start = time.time()
        embedding = embedder.create_embedding(str(file_path), features)
        elapsed = time.time() - start
        embedding_times.append(elapsed)
        print(f"   이미지 {i+1}: {elapsed:.4f}초 (임베딩 차원: {len(embedding)})")

    avg_embedding = sum(embedding_times) / len(embedding_times)
    print(f"\n   📊 평균 임베딩 시간: {avg_embedding:.4f}초")
    print(f"   📊 처리량: {1/avg_embedding:.2f} 이미지/초\n")

    # 4. DB 저장 성능 측정
    print("4️⃣  데이터베이스 저장 성능 측정")
    print("-" * 70)
    db_save_times = []

    for i, file_path in enumerate(face_files[:5]):
        start = time.time()
        success = system.processor.process_face_file(str(file_path))
        elapsed = time.time() - start

        if success:
            db_save_times.append(elapsed)
            print(f"   이미지 {i+1}: {elapsed:.4f}초")
        else:
            print(f"   이미지 {i+1}: {elapsed:.4f}초 (중복 스킵)")

    if db_save_times:
        avg_db_save = sum(db_save_times) / len(db_save_times)
        print(f"\n   📊 평균 DB 저장 시간: {avg_db_save:.4f}초")
        print(f"   📊 처리량: {1/avg_db_save:.2f} 이미지/초\n")
    else:
        print(f"\n   ⚠️  모든 이미지가 이미 DB에 존재함\n")

    # 5. 유사도 검색 성능 측정
    print("5️⃣  유사도 검색 (Vector Search) 성능 측정")
    print("-" * 70)

    # 쿼리용 임베딩 생성
    query_file = face_files[0]
    query_features = analyzer.analyze_face(str(query_file))
    query_embedding = embedder.create_embedding(str(query_file), query_features)

    search_times = []
    result_counts = [5, 10, 20, 50]

    for n_results in result_counts:
        start = time.time()
        results = system.db_manager.search_faces(query_embedding, n_results=n_results)
        elapsed = time.time() - start
        search_times.append(elapsed)
        print(f"   상위 {n_results}개 검색: {elapsed:.4f}초 (결과: {len(results)}개)")

    avg_search = sum(search_times) / len(search_times)
    print(f"\n   📊 평균 검색 시간: {avg_search:.4f}초")
    print(f"   📊 검색 처리량: {1/avg_search:.2f} 쿼리/초\n")

    # 6. 메타데이터 검색 성능 측정
    print("6️⃣  메타데이터 검색 성능 측정")
    print("-" * 70)

    metadata_filters = [
        {'skin_color': 'light'},
        {'age_group': 'adult'},
        {'hair_color': 'brown'}
    ]

    metadata_search_times = []

    for filter_dict in metadata_filters:
        start = time.time()
        results = system.db_manager.search_by_metadata(filter_dict, n_results=10)
        elapsed = time.time() - start
        metadata_search_times.append(elapsed)
        print(f"   필터 {filter_dict}: {elapsed:.4f}초 (결과: {len(results)}개)")

    if metadata_search_times:
        avg_metadata_search = sum(metadata_search_times) / len(metadata_search_times)
        print(f"\n   📊 평균 메타데이터 검색 시간: {avg_metadata_search:.4f}초")
        print(f"   📊 검색 처리량: {1/avg_metadata_search:.2f} 쿼리/초\n")

    # 7. 하이브리드 검색 성능 측정
    print("7️⃣  하이브리드 검색 (Vector + Metadata) 성능 측정")
    print("-" * 70)

    hybrid_search_times = []

    for filter_dict in metadata_filters:
        start = time.time()
        results = system.db_manager.hybrid_search(query_embedding, filter_dict, n_results=10)
        elapsed = time.time() - start
        hybrid_search_times.append(elapsed)
        print(f"   벡터 + {filter_dict}: {elapsed:.4f}초 (결과: {len(results)}개)")

    if hybrid_search_times:
        avg_hybrid_search = sum(hybrid_search_times) / len(hybrid_search_times)
        print(f"\n   📊 평균 하이브리드 검색 시간: {avg_hybrid_search:.4f}초")
        print(f"   📊 검색 처리량: {1/avg_hybrid_search:.2f} 쿼리/초\n")

    # 전체 요약
    print("=" * 70)
    print("  📊 성능 요약")
    print("=" * 70)
    print()
    print(f"  1. 이미지 다운로드:        {avg_download if download_times else 0:.3f}초 ({1/avg_download if download_times else 0:.2f} img/s)")
    print(f"  2. 특징 추출:             {avg_analysis:.4f}초 ({1/avg_analysis:.2f} img/s)")
    print(f"  3. 벡터 임베딩:           {avg_embedding:.4f}초 ({1/avg_embedding:.2f} img/s)")

    if db_save_times:
        print(f"  4. DB 저장:               {avg_db_save:.4f}초 ({1/avg_db_save:.2f} img/s)")
    else:
        print(f"  4. DB 저장:               N/A (중복)")

    print(f"  5. 벡터 검색:             {avg_search:.4f}초 ({1/avg_search:.2f} query/s)")
    print(f"  6. 메타데이터 검색:       {avg_metadata_search:.4f}초 ({1/avg_metadata_search:.2f} query/s)")
    print(f"  7. 하이브리드 검색:       {avg_hybrid_search:.4f}초 ({1/avg_hybrid_search:.2f} query/s)")
    print()

    # 전체 파이프라인 시간
    total_pipeline = avg_analysis + avg_embedding + (avg_db_save if db_save_times else avg_analysis)
    print(f"  📌 전체 파이프라인 (분석+임베딩+저장): {total_pipeline:.4f}초")
    print(f"  📌 전체 처리량: {1/total_pipeline:.2f} 이미지/초")
    print()

    # 시스템 통계
    stats = system.stats.get_stats()
    print("=" * 70)
    print("  📈 시스템 통계")
    print("=" * 70)
    print(f"  다운로드 시도: {stats['download_attempts']}")
    print(f"  다운로드 성공: {stats['download_success']}")
    print(f"  다운로드 중복: {stats['download_duplicates']}")
    print(f"  처리 완료: {stats['embed_processed']}")
    print(f"  임베딩 성공: {stats['embed_success']}")
    print(f"  검색 쿼리: {stats['search_queries']}")
    print()

    # DB 정보
    db_info = system.db_manager.get_collection_info()
    print(f"  DB 총 항목 수: {db_info.get('count', 0)}")
    print(f"  DB 경로: {db_info.get('path', 'N/A')}")
    print()
    print("=" * 70)

if __name__ == "__main__":
    benchmark_performance()
