"""Tests for the web crawler utility."""

import asyncio
from pathlib import Path
import pytest
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.crawler import WebCrawler


@pytest.mark.asyncio
async def test_crawler_initialization():
    """Test that crawler can be initialized."""
    crawler = WebCrawler(output_dir="data/data/test_crawler")
    assert crawler.output_dir == Path("data/data/test_crawler")
    assert crawler.timeout == 30
    assert crawler.max_retries == 3
    await crawler.close()


@pytest.mark.asyncio
async def test_clean_filename():
    """Test filename cleaning."""
    crawler = WebCrawler()
    
    # Test Vietnamese characters
    assert crawler._clean_filename("Hà Nội") == "Hà_Nội"
    
    # Test special characters
    assert crawler._clean_filename("Test/File:Name?") == "Test_File_Name"
    
    # Test multiple spaces
    assert crawler._clean_filename("Test   Multiple   Spaces") == "Test_Multiple_Spaces"
    
    await crawler.close()


@pytest.mark.asyncio
async def test_wikipedia_api_url_parsing():
    """Test that Wikipedia URLs are correctly parsed."""
    crawler = WebCrawler(output_dir="data/data/test_crawler")
    
    from urllib.parse import urlparse, unquote
    
    # Test URL parsing
    url = "https://vi.wikipedia.org/wiki/Hà_Nội"
    parsed = urlparse(url)
    lang = parsed.netloc.split('.')[0]
    path_parts = parsed.path.split('/')
    
    assert lang == "vi"
    assert len(path_parts) >= 3
    assert path_parts[1] == "wiki"
    
    page_title = unquote(path_parts[2])
    assert page_title == "Hà_Nội"
    
    await crawler.close()


@pytest.mark.asyncio
async def test_crawl_url_invalid():
    """Test that invalid URLs are handled gracefully."""
    crawler = WebCrawler(output_dir="data/data/test_crawler")
    
    # Test with invalid URL format
    result = await crawler.crawl_url(
        "https://vi.wikipedia.org/invalid",
        category="test"
    )
    
    # Should return None for failed crawls
    assert result is None
    
    await crawler.close()


def test_html_to_markdown_basic():
    """Test basic HTML to markdown conversion."""
    crawler = WebCrawler()
    
    from bs4 import BeautifulSoup
    
    html = """
    <div>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <h2>Subtitle</h2>
        <p>Another paragraph.</p>
    </div>
    """
    
    soup = BeautifulSoup(html, 'html.parser')
    markdown = crawler._html_to_markdown(soup)
    
    assert "Title" in markdown
    assert "This is a paragraph" in markdown
    assert "Subtitle" in markdown


if __name__ == '__main__':
    pytest.main([__file__, "-v"])

