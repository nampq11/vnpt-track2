"""Web crawler utility for extracting content and saving as markdown in .txt format.

This module provides functionality to crawl web pages (especially Wikipedia)
and save them in markdown format with .txt extension in the data/ folder.
"""

import asyncio
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup


class WebCrawler:
    """Crawler for extracting web content and converting to markdown."""
    
    def __init__(
        self,
        output_dir: str = "data/data",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """Initialize the crawler.
        
        Args:
            output_dir: Directory to save crawled data
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    async def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def _clean_filename(self, text: str) -> str:
        """Clean text to create valid filename.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned filename
        """
        # Replace invalid characters
        text = re.sub(r'[<>:"/\\|?*]', '_', text)
        # Replace multiple spaces/underscores with single underscore
        text = re.sub(r'[\s_]+', '_', text)
        # Remove leading/trailing underscores
        text = text.strip('_')
        return text
    
    def _html_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert HTML soup to markdown format.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Markdown formatted text
        """
        markdown_lines = []
        
        for element in soup.descendants:
            if element.name == 'h1':
                markdown_lines.append(f"\n=== {element.get_text().strip()} ===\n")
            elif element.name == 'h2':
                markdown_lines.append(f"\n  === {element.get_text().strip()} ===")
            elif element.name == 'h3':
                markdown_lines.append(f"\n    === {element.get_text().strip()} ===")
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"{text}\n")
            elif element.name == 'li':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"  - {text}")
            elif element.name == 'blockquote':
                text = element.get_text().strip()
                if text:
                    # Add quote formatting
                    quoted = '\n'.join(f'"{line}"' for line in text.split('\n') if line.strip())
                    markdown_lines.append(f"\n{quoted}\n")
        
        return '\n'.join(markdown_lines)
    
    def _fetch_wikipedia_via_api(self, title: str, lang: str = 'vi') -> Optional[tuple[str, str]]:
        """Fetch Wikipedia content via API.
        
        Args:
            title: Wikipedia page title
            lang: Language code (default: vi)
            
        Returns:
            Tuple of (title, content) or None
        """
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            'action': 'parse',
            'page': title,
            'format': 'json',
            'prop': 'text|displaytitle',
            'disableeditsection': 1,
            'disabletoc': 1,
        }
        
        try:
            response = self.session.get(api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' in data:
                title_html = data['parse']['displaytitle']
                # Clean HTML tags from title
                title_soup = BeautifulSoup(title_html, 'html.parser')
                title = title_soup.get_text().strip()
                html = data['parse']['text']['*']
                return title, html
        except Exception as e:
            print(f"⚠️  Wikipedia API failed: {e}")
        
        return None
    
    def _extract_wikipedia_content(self, html: str, url: str) -> tuple[str, str]:
        """Extract content from Wikipedia page.
        
        Args:
            html: HTML content
            url: Page URL
            
        Returns:
            Tuple of (title, markdown_content)
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1', class_='firstHeading') or soup.find('h1')
        title = title_elem.get_text().strip() if title_elem else "Untitled"
        
        # Find main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            content_div = soup.find('div', class_='mw-parser-output')
        
        if not content_div:
            return title, ""
        
        # Remove unwanted elements
        for unwanted in content_div.find_all(['script', 'style', 'sup', 'table', 'div.reflist']):
            unwanted.decompose()
        
        # Remove reference sections
        for ref_section in content_div.find_all(['span', 'div'], class_=re.compile(r'reference|citation|mw-editsection')):
            ref_section.decompose()
        
        # Extract paragraphs and headings
        markdown_content = self._html_to_markdown(content_div)
        
        return title, markdown_content
    
    def _extract_generic_content(self, html: str, url: str) -> tuple[str, str]:
        """Extract content from generic web page.
        
        Args:
            html: HTML content
            url: Page URL
            
        Returns:
            Tuple of (title, markdown_content)
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title_elem = soup.find('title') or soup.find('h1')
        title = title_elem.get_text().strip() if title_elem else "Untitled"
        
        # Remove unwanted elements
        for unwanted in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            unwanted.decompose()
        
        # Find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if not main_content:
            return title, ""
        
        markdown_content = self._html_to_markdown(main_content)
        
        return title, markdown_content
    
    async def crawl_url(
        self,
        url: str,
        category: Optional[str] = None,
        force: bool = False
    ) -> Optional[Path]:
        """Crawl a single URL and save as markdown in .txt file.
        
        Args:
            url: URL to crawl
            category: Category subfolder (optional)
            force: Overwrite existing files
            
        Returns:
            Path to saved file or None if failed
        """
        print(f"Crawling: {url}")
        
        # Detect if Wikipedia and try API first
        is_wikipedia = 'wikipedia.org' in url
        
        if is_wikipedia:
            # Extract title and language from URL
            # Example: https://vi.wikipedia.org/wiki/Đà_Nẵng
            parsed = urlparse(url)
            lang = parsed.netloc.split('.')[0]
            path_parts = parsed.path.split('/')
            if len(path_parts) >= 3 and path_parts[1] == 'wiki':
                page_title = unquote(path_parts[2])
                
                # Try Wikipedia API first
                api_result = self._fetch_wikipedia_via_api(page_title, lang)
                if api_result:
                    title, html_content = api_result
                    content = self._html_to_markdown(BeautifulSoup(html_content, 'html.parser'))
                else:
                    # Fallback to direct HTTP
                    print("⚠️  Falling back to direct HTTP...")
                    for attempt in range(self.max_retries):
                        try:
                            response = self.session.get(url, timeout=self.timeout)
                            response.raise_for_status()
                            title, content = self._extract_wikipedia_content(response.text, url)
                            break
                        except Exception as e:
                            if attempt == self.max_retries - 1:
                                print(f"❌ Failed to crawl {url}: {e}")
                                return None
                            await asyncio.sleep(2 ** attempt)
            else:
                print(f"❌ Invalid Wikipedia URL format: {url}")
                return None
        else:
            # Non-Wikipedia URL
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    title, content = self._extract_generic_content(response.text, url)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"❌ Failed to crawl {url}: {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)
        
        if not content.strip():
            print(f"⚠️  No content extracted from {url}")
            return None
        
        # Create output directory
        if category:
            output_path = self.output_dir / self._clean_filename(category)
        else:
            # Try to extract category from URL or use domain
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            if len(path_parts) > 1:
                output_path = self.output_dir / self._clean_filename(path_parts[-2])
            else:
                output_path = self.output_dir / self._clean_filename(parsed.netloc)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename from title
        filename = self._clean_filename(title) + '.txt'
        filepath = output_path / filename
        
        # Check if file exists
        if filepath.exists() and not force:
            print(f"⚠️  File already exists: {filepath}")
            return filepath
        
        # Format content with metadata
        formatted_content = f"""Tiêu đề: {title}
URL: {url}
--------------------
{content}
"""
        
        # Save file
        filepath.write_text(formatted_content, encoding='utf-8')
        print(f"✅ Saved: {filepath}")
        
        return filepath
    
    async def crawl_urls(
        self,
        urls: List[str],
        category: Optional[str] = None,
        force: bool = False,
        delay: float = 1.0
    ) -> List[Path]:
        """Crawl multiple URLs with rate limiting.
        
        Args:
            urls: List of URLs to crawl
            category: Category subfolder (optional)
            force: Overwrite existing files
            delay: Delay between requests in seconds
            
        Returns:
            List of paths to saved files
        """
        results = []
        
        for i, url in enumerate(urls):
            result = await self.crawl_url(url, category, force)
            if result:
                results.append(result)
            
            # Add delay between requests (except for last one)
            if i < len(urls) - 1:
                await asyncio.sleep(delay)
        
        return results
    
    async def crawl_from_file(
        self,
        filepath: str,
        category: Optional[str] = None,
        force: bool = False,
        delay: float = 1.0
    ) -> List[Path]:
        """Crawl URLs from a text file (one URL per line).
        
        Args:
            filepath: Path to file containing URLs
            category: Category subfolder (optional)
            force: Overwrite existing files
            delay: Delay between requests in seconds
            
        Returns:
            List of paths to saved files
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Found {len(urls)} URLs to crawl")
        return await self.crawl_urls(urls, category, force, delay)


async def main():
    """Main entry point for testing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python crawler.py <url> [category]")
        sys.exit(1)
    
    url = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else None
    
    crawler = WebCrawler()
    try:
        await crawler.crawl_url(url, category)
    finally:
        await crawler.close()


if __name__ == '__main__':
    asyncio.run(main())

