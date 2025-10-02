import unittest
import asyncio
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from main import (
    NewsArticle, NewsMetadata, NewsChunk,
    EnhancedNaverNewsAPI, EnhancedPromptManager, EnhancedChromaDBManager,
    EnhancedClaudeClient, EnhancedNewsCollector, EnhancedNewsWriter,
    EnhancedAINewsWriterSystem
)


class TestNewsArticle(unittest.TestCase):
    """Test NewsArticle dataclass"""

    def test_news_article_creation(self):
        article = NewsArticle(
            title="Test Title",
            link="http://test.com",
            description="Test Description",
            pub_date="Mon, 01 Jan 2024 10:00:00 +0900",
            content="Test Content"
        )

        self.assertEqual(article.title, "Test Title")
        self.assertEqual(article.link, "http://test.com")
        self.assertEqual(article.description, "Test Description")
        self.assertEqual(article.pub_date, "Mon, 01 Jan 2024 10:00:00 +0900")
        self.assertEqual(article.content, "Test Content")

    def test_news_article_default_content(self):
        article = NewsArticle(
            title="Test Title",
            link="http://test.com",
            description="Test Description",
            pub_date="Mon, 01 Jan 2024 10:00:00 +0900"
        )

        self.assertEqual(article.content, "")


class TestNewsMetadata(unittest.TestCase):
    """Test NewsMetadata dataclass"""

    def test_news_metadata_creation(self):
        metadata = NewsMetadata(
            relevance_score=8,
            topics=["tech", "business"],
            keywords=["AI", "ML"],
            summary="Test summary",
            sentiment="positive",
            importance=7,
            company_mentions=["TestCorp"],
            date="2024-01-01",
            source="test_source"
        )

        self.assertEqual(metadata.relevance_score, 8)
        self.assertEqual(metadata.topics, ["tech", "business"])
        self.assertEqual(metadata.keywords, ["AI", "ML"])
        self.assertEqual(metadata.summary, "Test summary")
        self.assertEqual(metadata.sentiment, "positive")
        self.assertEqual(metadata.importance, 7)
        self.assertEqual(metadata.company_mentions, ["TestCorp"])
        self.assertEqual(metadata.date, "2024-01-01")
        self.assertEqual(metadata.source, "test_source")


class TestNewsChunk(unittest.TestCase):
    """Test NewsChunk dataclass"""

    def test_news_chunk_creation(self):
        chunk = NewsChunk(
            chunk_id=1,
            content="Test chunk content",
            topics=["topic1"],
            keywords=["keyword1"],
            chunk_type="title"
        )

        self.assertEqual(chunk.chunk_id, 1)
        self.assertEqual(chunk.content, "Test chunk content")
        self.assertEqual(chunk.topics, ["topic1"])
        self.assertEqual(chunk.keywords, ["keyword1"])
        self.assertEqual(chunk.chunk_type, "title")


class TestEnhancedNaverNewsAPI(unittest.TestCase):
    """Test EnhancedNaverNewsAPI class"""

    def setUp(self):
        self.api = EnhancedNaverNewsAPI("test_id", "test_secret")

    def test_init_with_real_keys(self):
        api = EnhancedNaverNewsAPI("real_id", "real_secret")
        self.assertEqual(api.client_id, "real_id")
        self.assertEqual(api.client_secret, "real_secret")
        self.assertFalse(api.test_mode)

    def test_init_with_dummy_keys(self):
        api = EnhancedNaverNewsAPI("YOUR_NAVER_CLIENT_ID", "test_secret")
        self.assertTrue(api.test_mode)

    def test_clean_html(self):
        html_text = "<p>Test <b>bold</b> text</p>"
        cleaned = self.api._clean_html(html_text)
        self.assertEqual(cleaned, "Test bold text")

    def test_get_dummy_news(self):
        dummy_news = self.api._get_dummy_news("삼성전자", 3)
        self.assertEqual(len(dummy_news), 3)

        for article in dummy_news:
            self.assertIsInstance(article, NewsArticle)
            self.assertIn("삼성전자", article.title)
            self.assertTrue(article.link.startswith("http://test.com/news"))

    @patch('requests.get')
    def test_search_news_success(self, mock_get):
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'items': [
                {
                    'title': 'Test News Title',
                    'link': 'http://test.com/news1',
                    'description': 'Test description',
                    'pubDate': 'Mon, 01 Jan 2024 10:00:00 +0900'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock _fetch_article_content to avoid actual web requests
        with patch.object(self.api, '_fetch_article_content', return_value="Test content"):
            api_real = EnhancedNaverNewsAPI("real_id", "real_secret")
            results = api_real.search_news("test query")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Test News Title")
            self.assertEqual(results[0].content, "Test content")

    @patch('requests.get')
    def test_search_news_failure(self, mock_get):
        # Mock API failure
        mock_get.side_effect = Exception("API Error")

        api_real = EnhancedNaverNewsAPI("real_id", "real_secret")
        results = api_real.search_news("test query")

        # Should return dummy news on failure
        self.assertGreater(len(results), 0)
        self.assertIn("test", results[0].title.lower())

    def test_search_news_with_keywords(self):
        # Test with test mode (no actual API calls)
        results = self.api.search_news_with_keywords(
            "삼성전자",
            ["스마트폰", "갤럭시"],
            display=5
        )

        self.assertLessEqual(len(results), 5)
        for article in results:
            self.assertIn("삼성전자", article.title)


class TestEnhancedPromptManager(unittest.TestCase):
    """Test EnhancedPromptManager class"""

    def test_get_news_analysis_prompt(self):
        prompt = EnhancedPromptManager.get_news_analysis_prompt(
            "Test news content", "TestCorp"
        )

        self.assertIn("TestCorp", prompt)
        self.assertIn("Test news content", prompt)
        self.assertIn("relevance_score", prompt)
        self.assertIn("json", prompt.lower())

    def test_get_news_chunking_prompt(self):
        prompt = EnhancedPromptManager.get_news_chunking_prompt("Test news content")

        self.assertIn("Test news content", prompt)
        self.assertIn("청킹 규칙", prompt)
        self.assertIn("json", prompt.lower())

    def test_get_enhanced_news_generation_prompt(self):
        prompt = EnhancedPromptManager.get_enhanced_news_generation_prompt(
            "Technology", ["AI", "ML"], "Test facts", "Reference materials"
        )

        self.assertIn("Technology", prompt)
        self.assertIn("AI", prompt)
        self.assertIn("Test facts", prompt)
        self.assertIn("Reference materials", prompt)

    def test_get_enhanced_news_generation_prompt_with_length(self):
        prompt = EnhancedPromptManager.get_enhanced_news_generation_prompt(
            "Technology", ["AI"], "Test facts", "Reference", "50줄 수"
        )

        self.assertIn("50줄", prompt)
        self.assertIn("정확히 50줄", prompt)

    def test_get_quality_check_prompt(self):
        prompt = EnhancedPromptManager.get_quality_check_prompt("Test news content")

        self.assertIn("Test news content", prompt)
        self.assertIn("품질", prompt)
        self.assertIn("overall_score", prompt)


class TestEnhancedChromaDBManager(unittest.TestCase):
    """Test EnhancedChromaDBManager class"""

    def setUp(self):
        # Use temporary directory for test database
        self.test_db_path = tempfile.mkdtemp()
        self.db_manager = EnhancedChromaDBManager(self.test_db_path)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_db_path, ignore_errors=True)

    def test_init(self):
        self.assertIsNotNone(self.db_manager.client)
        self.assertIsNotNone(self.db_manager.collection)
        self.assertIsNotNone(self.db_manager.embedding_function)

    def test_store_news_chunk(self):
        chunk = NewsChunk(
            chunk_id=1,
            content="Test chunk content",
            topics=["test"],
            keywords=["keyword"],
            chunk_type="test"
        )

        metadata = NewsMetadata(
            relevance_score=8,
            topics=["test"],
            keywords=["keyword"],
            summary="Test summary",
            sentiment="neutral",
            importance=7,
            company_mentions=["TestCorp"],
            date="2024-01-01",
            source="test"
        )

        embedding = [0.1] * 768

        # Should not raise an exception
        self.db_manager.store_news_chunk(chunk, metadata, embedding)

        # Verify storage by checking collection count
        count = self.db_manager.collection.count()
        self.assertGreater(count, 0)

    def test_get_collection_stats(self):
        stats = self.db_manager.get_collection_stats()

        self.assertIn("total_chunks", stats)
        self.assertIn("collection_name", stats)
        self.assertIn("embedding_dimension", stats)

    def test_search_relevant_news_empty_collection(self):
        results = self.db_manager.search_relevant_news("test query")

        self.assertIn("documents", results)
        self.assertIn("metadatas", results)
        self.assertIn("distances", results)


class TestEnhancedClaudeClient(unittest.TestCase):
    """Test EnhancedClaudeClient class"""

    def setUp(self):
        self.client = EnhancedClaudeClient("test_api_key")

    def test_init_with_api_key(self):
        client = EnhancedClaudeClient("real_api_key")
        self.assertEqual(client.api_key, "real_api_key")
        self.assertIsNotNone(client.client)

    def test_init_without_api_key(self):
        client = EnhancedClaudeClient()
        self.assertIsNone(client.client)

    def test_get_enhanced_dummy_response_news_analysis(self):
        prompt = "뉴스 분석 prompt with 키워드: 삼성전자, AI"
        response = self.client._get_enhanced_dummy_response(prompt)

        self.assertIn("relevance_score", response)
        self.assertIn("삼성전자", response)
        # Should be valid JSON
        json.loads(response)

    def test_get_enhanced_dummy_response_chunking(self):
        prompt = "청크로 분할 prompt with 키워드: 삼성전자"
        response = self.client._get_enhanced_dummy_response(prompt)

        self.assertIn("chunks", response)
        self.assertIn("삼성전자", response)
        # Should be valid JSON
        json.loads(response)

    def test_get_enhanced_dummy_response_quality_check(self):
        prompt = "품질을 평가 prompt"
        response = self.client._get_enhanced_dummy_response(prompt)

        self.assertIn("overall_score", response)
        self.assertIn("approval", response)
        # Should be valid JSON
        json.loads(response)

    def test_get_enhanced_dummy_response_news_generation(self):
        prompt = "뉴스 생성 prompt with 키워드: 삼성전자, AI, 스마트폰"
        response = self.client._get_enhanced_dummy_response(prompt)

        self.assertIn("삼성전자", response)
        self.assertIn("AI", response)
        self.assertNotIn("{", response)  # Should not be JSON for news generation


class TestEnhancedNewsCollectorAsync(unittest.TestCase):
    """Test EnhancedNewsCollector class (async methods)"""

    def setUp(self):
        self.claude_client = EnhancedClaudeClient()
        self.db_manager = EnhancedChromaDBManager(tempfile.mkdtemp())
        self.naver_api = EnhancedNaverNewsAPI("test_id", "test_secret")
        self.collector = EnhancedNewsCollector(
            self.claude_client, self.db_manager, self.naver_api
        )

    def test_init(self):
        self.assertEqual(self.collector.claude_client, self.claude_client)
        self.assertEqual(self.collector.db_manager, self.db_manager)
        self.assertEqual(self.collector.naver_api, self.naver_api)

    def test_is_recent_article(self):
        # Test with recent date
        recent_date = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
        if not recent_date.endswith(" +0900"):
            recent_date = recent_date.rsplit(" ", 1)[0] + " +0900"

        self.assertTrue(self.collector._is_recent_article(recent_date, 365))

        # Test with invalid date (should return True)
        self.assertTrue(self.collector._is_recent_article("invalid_date", 365))

    def test_convert_pub_date(self):
        pub_date = "Mon, 01 Jan 2024 10:00:00 +0900"
        converted = self.collector._convert_pub_date(pub_date)
        self.assertEqual(converted, "2024-01-01")

        # Test with invalid date
        invalid_converted = self.collector._convert_pub_date("invalid")
        self.assertIsInstance(invalid_converted, str)
        self.assertEqual(len(invalid_converted), 10)  # YYYY-MM-DD format

    def test_extract_json_from_response(self):
        # Test with JSON block
        response_with_block = '''
        Some text before
        ```json
        {"key": "value", "number": 42}
        ```
        Some text after
        '''

        result = self.collector._extract_json_from_response(response_with_block)
        self.assertEqual(result, {"key": "value", "number": 42})

        # Test with inline JSON
        response_inline = 'Text before {"key": "value"} text after'
        result = self.collector._extract_json_from_response(response_inline)
        self.assertEqual(result, {"key": "value"})


class TestEnhancedNewsWriterAsync(unittest.TestCase):
    """Test EnhancedNewsWriter class (async methods)"""

    def setUp(self):
        self.claude_client = EnhancedClaudeClient()
        self.db_manager = EnhancedChromaDBManager(tempfile.mkdtemp())
        self.writer = EnhancedNewsWriter(self.claude_client, self.db_manager)

    def test_init(self):
        self.assertEqual(self.writer.claude_client, self.claude_client)
        self.assertEqual(self.writer.db_manager, self.db_manager)

    def test_get_full_generation_prompt(self):
        prompt = self.writer.get_full_generation_prompt(
            "Technology", ["AI"], "Test facts", "Reference materials", "50줄 수"
        )

        self.assertIsInstance(prompt, str)
        self.assertIn("Technology", prompt)
        self.assertIn("AI", prompt)
        self.assertIn("50줄", prompt)

    def test_build_comprehensive_reference_materials(self):
        # Test with empty results
        empty_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        materials = self.writer._build_comprehensive_reference_materials(empty_results)
        self.assertIn("관련 참고 자료가 없습니다", materials)

        # Test with sample results
        sample_results = {
            'documents': [['Test document content']],
            'metadatas': [[{
                'topics': '["tech"]',
                'keywords': '["AI"]',
                'company_mentions': '["TestCorp"]',
                'source': 'test_source',
                'date': '2024-01-01',
                'importance': 8,
                'relevance_score': 9,
                'sentiment': 'positive',
                'summary': 'Test summary'
            }]],
            'distances': [[0.1]]
        }

        materials = self.writer._build_comprehensive_reference_materials(sample_results)
        self.assertIn("Test document content", materials)
        self.assertIn("TestCorp", materials)
        self.assertIn("test_source", materials)


class TestEnhancedAINewsWriterSystemAsync(unittest.TestCase):
    """Test EnhancedAINewsWriterSystem class (async methods)"""

    def setUp(self):
        self.system = EnhancedAINewsWriterSystem(
            claude_api_key="test_key",
            naver_client_id="test_id",
            naver_client_secret="test_secret",
            db_path=tempfile.mkdtemp()
        )

    def test_init(self):
        self.assertIsNotNone(self.system.claude_client)
        self.assertIsNotNone(self.system.db_manager)
        self.assertIsNotNone(self.system.naver_api)
        self.assertIsNotNone(self.system.news_collector)
        self.assertIsNotNone(self.system.news_writer)

    def test_get_system_stats(self):
        stats = self.system.get_system_stats()

        self.assertIn("database", stats)
        self.assertIn("api_requests", stats)
        self.assertIn("naver_test_mode", stats)


# Async test methods require special handling
class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


class TestAsyncMethods(AsyncTestCase):
    """Test async methods of various classes"""

    def setUp(self):
        super().setUp()
        self.claude_client = EnhancedClaudeClient()
        self.db_manager = EnhancedChromaDBManager(tempfile.mkdtemp())
        self.naver_api = EnhancedNaverNewsAPI("test_id", "test_secret")
        self.collector = EnhancedNewsCollector(
            self.claude_client, self.db_manager, self.naver_api
        )
        self.writer = EnhancedNewsWriter(self.claude_client, self.db_manager)
        self.system = EnhancedAINewsWriterSystem(db_path=tempfile.mkdtemp())

    def test_claude_client_generate_response(self):
        """Test EnhancedClaudeClient.generate_response"""
        result = self.run_async(
            self.claude_client.generate_response("Test prompt", max_tokens=100)
        )

        self.assertIn("elapsed", result)
        self.assertIn("model", result)
        self.assertIn("prompt", result)
        self.assertIn("response", result)
        self.assertEqual(result["prompt"], "Test prompt")

    def test_collector_collect_and_store_news(self):
        """Test EnhancedNewsCollector.collect_and_store_news"""
        article = NewsArticle(
            title="삼성전자 Test Article",
            link="http://test.com",
            description="Test description about 삼성전자",
            pub_date="Mon, 01 Jan 2024 10:00:00 +0900",
            content="Test content with 삼성전자 mention multiple times. 삼성전자 is testing."
        )

        result = self.run_async(
            self.collector.collect_and_store_news("삼성전자", article)
        )

        self.assertTrue(result)

    def test_collector_collect_and_store_news_low_relevance(self):
        """Test EnhancedNewsCollector.collect_and_store_news with low relevance"""
        article = NewsArticle(
            title="Unrelated Article",
            link="http://test.com",
            description="This has nothing to do with the company",
            pub_date="Mon, 01 Jan 2024 10:00:00 +0900",
            content="No company mentions here"
        )

        result = self.run_async(
            self.collector.collect_and_store_news("삼성전자", article)
        )

        self.assertFalse(result)

    def test_writer_generate_enhanced_news(self):
        """Test EnhancedNewsWriter.generate_enhanced_news"""
        result = self.run_async(
            self.writer.generate_enhanced_news(
                topic="Technology News",
                keywords=["삼성전자", "AI"],
                user_facts="Test facts about technology",
                style="기업 보도형",
                use_rag=False  # Disable RAG to avoid database dependencies
            )
        )

        self.assertIn("response", result)
        self.assertIn("elapsed", result)
        self.assertIn("model", result)

    def test_system_collect_manual_news(self):
        """Test EnhancedAINewsWriterSystem.collect_manual_news"""
        result = self.run_async(
            self.system.collect_manual_news(
                "삼성전자",
                "삼성전자가 새로운 기술을 발표했습니다. 이는 혁신적인 발전입니다."
            )
        )

        self.assertTrue(result)

    def test_system_write_news(self):
        """Test EnhancedAINewsWriterSystem.write_news"""
        result = self.run_async(
            self.system.write_news(
                topic="Tech Innovation",
                keywords=["삼성전자", "Innovation"],
                user_facts="Test facts",
                use_rag=False
            )
        )

        self.assertIsNotNone(result)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)