#!/usr/bin/env python3
"""
📊 Status Manager
Comprehensive status tracking and reporting for document processing
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class UrlStatus:
    """📌 URL 요청 상태 정보"""
    url: str
    requested_at: str
    documents_found: int
    total_size_mb: float
    status: str  # 'scanned', 'downloading', 'completed', 'failed'
    
@dataclass
class DownloadStatus:
    """📥 다운로드 상태 정보"""
    total_files: int
    downloaded_files: int
    failed_files: int
    total_size_mb: float
    download_directory: str
    file_list: List[str]
    last_updated: str

@dataclass
class ChunkStatus:
    """🧩 청킹 상태 정보"""
    total_chunks: int
    total_characters: int
    files_processed: int
    processing_errors: List[str]
    chunk_size: int
    overlap: int
    semantic_chunking: bool
    last_updated: str

@dataclass
class VectorStatus:
    """🗄️ 벡터 데이터베이스 상태 정보"""
    is_ready: bool
    collection_name: str
    vector_count: int
    vector_dimensions: int
    index_size_mb: float
    embedding_model: str
    last_indexed: str
    search_capabilities: List[str]

class StatusManager:
    """📊 전체 상태 관리자"""
    
    def __init__(self, status_file: str = "processing_status.json"):
        # 절대 경로로 저장하여 어디서 실행해도 같은 파일 사용
        self.status_file = os.path.abspath(status_file)
        self.current_status = {
            'url_history': [],
            'download_status': None,
            'chunk_status': None,
            'vector_status': None,
            'session_start': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        self._load_status()
        print(f"📁 Status file: {self.status_file}")
    
    def _load_status(self):
        """💾 저장된 상태 불러오기"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    saved_status = json.load(f)
                    self.current_status.update(saved_status)
        except Exception as e:
            logger.warning(f"상태 파일 로드 실패: {e}")
    
    def _save_status(self):
        """💾 현재 상태 저장"""
        try:
            self.current_status['last_updated'] = datetime.now().isoformat()
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"상태 파일 저장 실패: {e}")
    
    def update_url_status(self, url: str, documents_found: int, 
                         total_size_mb: float, status: str = 'scanned'):
        """📌 URL 스캔 상태 업데이트"""
        url_status = UrlStatus(
            url=url,
            requested_at=datetime.now().isoformat(),
            documents_found=documents_found,
            total_size_mb=total_size_mb,
            status=status
        )
        
        # 기존 URL이 있으면 업데이트, 없으면 추가
        url_history = self.current_status['url_history']
        for i, existing in enumerate(url_history):
            if existing.get('url') == url:
                url_history[i] = asdict(url_status)
                break
        else:
            url_history.append(asdict(url_status))
        
        # 최근 5개만 유지
        self.current_status['url_history'] = url_history[-5:]
        self._save_status()
    
    def update_download_status(self, total_files: int, downloaded_files: int, 
                             failed_files: int, total_size_mb: float,
                             download_directory: str, file_list: List[str]):
        """📥 다운로드 상태 업데이트"""
        download_status = DownloadStatus(
            total_files=total_files,
            downloaded_files=downloaded_files,
            failed_files=failed_files,
            total_size_mb=total_size_mb,
            download_directory=download_directory,
            file_list=file_list,
            last_updated=datetime.now().isoformat()
        )
        
        self.current_status['download_status'] = asdict(download_status)
        self._save_status()
    
    def update_chunk_status(self, total_chunks: int, total_characters: int,
                           files_processed: int, processing_errors: List[str],
                           chunk_size: int, overlap: int, semantic_chunking: bool):
        """🧩 청킹 상태 업데이트"""
        chunk_status = ChunkStatus(
            total_chunks=total_chunks,
            total_characters=total_characters,
            files_processed=files_processed,
            processing_errors=processing_errors,
            chunk_size=chunk_size,
            overlap=overlap,
            semantic_chunking=semantic_chunking,
            last_updated=datetime.now().isoformat()
        )
        
        self.current_status['chunk_status'] = asdict(chunk_status)
        self._save_status()
    
    def update_vector_status(self, is_ready: bool, collection_name: str = "",
                           vector_count: int = 0, vector_dimensions: int = 0,
                           index_size_mb: float = 0, embedding_model: str = "",
                           search_capabilities: List[str] = None):
        """🗄️ 벡터 데이터베이스 상태 업데이트"""
        vector_status = VectorStatus(
            is_ready=is_ready,
            collection_name=collection_name,
            vector_count=vector_count,
            vector_dimensions=vector_dimensions,
            index_size_mb=index_size_mb,
            embedding_model=embedding_model,
            last_indexed=datetime.now().isoformat() if is_ready else "",
            search_capabilities=search_capabilities or []
        )
        
        self.current_status['vector_status'] = asdict(vector_status)
        self._save_status()
    
    def get_comprehensive_status(self) -> str:
        """📊 종합 상태 보고서 생성"""
        status = self.current_status
        report = []
        
        # 📈 세션 정보
        report.append("🚀 **Document Processing Session Status**")
        report.append(f"🕐 Session started: {self._format_datetime(status.get('session_start', ''))}")
        report.append(f"🔄 Last updated: {self._format_datetime(status.get('last_updated', ''))}")
        report.append("")
        
        # 📌 URL 히스토리
        report.append("📌 **Recent URL Requests**")
        url_history = status.get('url_history', [])
        if url_history:
            for i, url_info in enumerate(reversed(url_history[-3:]), 1):
                report.append(f"   {i}. 🌐 {url_info['url']}")
                report.append(f"      📄 Found: {url_info['documents_found']} documents")
                report.append(f"      💾 Size: {url_info['total_size_mb']:.1f} MB")
                report.append(f"      ⏰ Requested: {self._format_datetime(url_info['requested_at'])}")
                report.append(f"      📊 Status: {self._get_status_emoji(url_info['status'])} {url_info['status']}")
                report.append("")
        else:
            report.append("   ❌ No URLs requested yet")
            report.append("")
        
        # 📥 다운로드 상태
        report.append("📥 **Download Status**")
        download_status = status.get('download_status')
        if download_status:
            report.append(f"   📊 Progress: {download_status['downloaded_files']}/{download_status['total_files']} files")
            if download_status['failed_files'] > 0:
                report.append(f"   ❌ Failed: {download_status['failed_files']} files")
            report.append(f"   💾 Total size: {download_status['total_size_mb']:.1f} MB")
            report.append(f"   📁 Directory: {download_status['download_directory']}")
            report.append(f"   🕐 Last update: {self._format_datetime(download_status['last_updated'])}")
            
            # 파일 목록 (최근 5개)
            file_list = download_status.get('file_list', [])
            if file_list:
                report.append("   📋 **Recent Files:**")
                for file_path in file_list[-5:]:
                    filename = os.path.basename(file_path)
                    report.append(f"      • {filename}")
                if len(file_list) > 5:
                    report.append(f"      ... and {len(file_list) - 5} more files")
        else:
            report.append("   ❌ No downloads completed yet")
        report.append("")
        
        # 🧩 청킹 상태
        report.append("🧩 **Chunking Status**")
        chunk_status = status.get('chunk_status')
        if chunk_status:
            report.append(f"   📊 Chunks created: {chunk_status['total_chunks']:,}")
            report.append(f"   📄 Files processed: {chunk_status['files_processed']}")
            report.append(f"   📝 Total characters: {chunk_status['total_characters']:,}")
            report.append(f"   ⚙️  Chunk size: {chunk_status['chunk_size']} characters")
            report.append(f"   🔗 Overlap: {chunk_status['overlap']} characters")
            report.append(f"   🧠 Semantic chunking: {'✅ Enabled' if chunk_status['semantic_chunking'] else '❌ Disabled'}")
            report.append(f"   🕐 Last update: {self._format_datetime(chunk_status['last_updated'])}")
            
            # 처리 오류
            errors = chunk_status.get('processing_errors', [])
            if errors:
                report.append(f"   ⚠️  Processing errors: {len(errors)}")
                for error in errors[-3:]:  # 최근 3개 오류만 표시
                    report.append(f"      • {error}")
        else:
            report.append("   ❌ No chunking completed yet")
        report.append("")
        
        # 🗄️ 벡터 데이터베이스 상태
        report.append("🗄️ **Vector Database Status**")
        vector_status = status.get('vector_status')
        if vector_status and vector_status['is_ready']:
            report.append(f"   ✅ Status: Ready for search")
            report.append(f"   📊 Collection: {vector_status['collection_name']}")
            report.append(f"   🔢 Vector count: {vector_status['vector_count']:,}")
            report.append(f"   📐 Dimensions: {vector_status['vector_dimensions']}")
            report.append(f"   💾 Index size: {vector_status['index_size_mb']:.1f} MB")
            report.append(f"   🤖 Model: {vector_status['embedding_model']}")
            report.append(f"   🕐 Last indexed: {self._format_datetime(vector_status['last_indexed'])}")
            
            capabilities = vector_status.get('search_capabilities', [])
            if capabilities:
                report.append(f"   🔍 Search capabilities: {', '.join(capabilities)}")
        else:
            report.append("   ❌ Vector database not ready")
        report.append("")
        
        # 💡 다음 단계 제안
        report.append("💡 **Suggested Next Steps**")
        next_steps = self._get_next_steps()
        for step in next_steps:
            report.append(f"   • {step}")
        
        return "\n".join(report)
    
    def get_quick_status(self) -> str:
        """⚡ 간단한 상태 요약"""
        status = self.current_status
        
        # 최근 URL
        recent_url = "None"
        url_history = status.get('url_history', [])
        if url_history:
            recent_url = url_history[-1]['url']
        
        # 다운로드 상태
        download_info = "No downloads"
        download_status = status.get('download_status')
        if download_status:
            download_info = f"{download_status['downloaded_files']}/{download_status['total_files']} files"
        
        # 청킹 상태
        chunk_info = "No chunks"
        chunk_status = status.get('chunk_status')
        if chunk_status:
            chunk_info = f"{chunk_status['total_chunks']:,} chunks"
        
        # 벡터 상태
        vector_info = "❌ Not ready"
        vector_status = status.get('vector_status')
        if vector_status and vector_status['is_ready']:
            vector_info = f"✅ {vector_status['vector_count']:,} vectors"
        
        return f"""⚡ **Quick Status**
🌐 Last URL: {recent_url}
📥 Downloaded: {download_info}
🧩 Chunked: {chunk_info}
🗄️ Vector DB: {vector_info}"""
    
    def _format_datetime(self, iso_string: str) -> str:
        """📅 날짜/시간 포맷팅"""
        try:
            if iso_string:
                dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        return "Unknown"
    
    def _get_status_emoji(self, status: str) -> str:
        """📊 상태별 이모지 반환"""
        emoji_map = {
            'scanned': '🔍',
            'downloading': '📥',
            'completed': '✅',
            'failed': '❌',
            'processing': '⚙️'
        }
        return emoji_map.get(status, '❓')
    
    def _get_next_steps(self) -> List[str]:
        """💡 다음 단계 제안"""
        steps = []
        status = self.current_status
        
        # URL 스캔되지 않은 경우
        if not status.get('url_history'):
            steps.append("🌐 Start by scanning a URL for documents")
            return steps
        
        # 다운로드되지 않은 경우
        download_status = status.get('download_status')
        if not download_status or download_status['downloaded_files'] == 0:
            steps.append("📥 Download documents from the scanned URL")
            return steps
        
        # 청킹되지 않은 경우
        chunk_status = status.get('chunk_status')
        if not chunk_status or chunk_status['total_chunks'] == 0:
            steps.append("🧩 Process downloaded files into chunks")
            return steps
        
        # 벡터 인덱싱되지 않은 경우
        vector_status = status.get('vector_status')
        if not vector_status or not vector_status['is_ready']:
            steps.append("🗄️ Index chunks into vector database")
            return steps
        
        # 모든 단계 완료
        steps.append("🔍 Ready to search documents!")
        steps.append("📊 Check processing statistics")
        steps.append("🌐 Process additional URLs")
        
        return steps
    
    def clear_status(self):
        """🗑️ 상태 초기화"""
        self.current_status = {
            'url_history': [],
            'download_status': None,
            'chunk_status': None,
            'vector_status': None,
            'session_start': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        self._save_status()
    
    def export_status(self, filename: str = None) -> str:
        """📤 상태 내보내기"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"status_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_status, f, ensure_ascii=False, indent=2)
            return f"📁 Status exported to {filename}"
        except Exception as e:
            return f"❌ Export failed: {e}"