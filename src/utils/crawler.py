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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
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
        processed_elements = set()  # Track processed elements to avoid duplicates
        
        # Process elements in document order (direct children first)
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'ul', 'ol', 'pre', 'div']):
            # Skip if already processed as part of parent
            if id(element) in processed_elements:
                continue
                
            if element.name == 'h1':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"\n=== {text} ===\n")
                    processed_elements.add(id(element))
                    
            elif element.name == 'h2':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"\n  === {text} ===\n")
                    processed_elements.add(id(element))
                    
            elif element.name == 'h3':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"\n    === {text} ===\n")
                    processed_elements.add(id(element))
                    
            elif element.name in ['h4', 'h5', 'h6']:
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"\n      === {text} ===\n")
                    processed_elements.add(id(element))
                    
            elif element.name == 'p':
                # Skip if this p is inside a blockquote or list (will be processed with parent)
                if element.find_parent(['blockquote', 'li']):
                    continue
                text = element.get_text().strip()
                if text and len(text) > 10:  # Skip very short paragraphs (likely navigation)
                    markdown_lines.append(f"{text}\n")
                    processed_elements.add(id(element))
                    
            elif element.name == 'blockquote':
                text = element.get_text().strip()
                if text:
                    # Add quote formatting with proper indentation
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    for line in lines:
                        markdown_lines.append(f"> {line}")
                    markdown_lines.append("")  # Add blank line after quote
                    processed_elements.add(id(element))
                    
            elif element.name in ['ul', 'ol']:
                # Process list items
                for li in element.find_all('li', recursive=False):
                    text = li.get_text().strip()
                    if text:
                        markdown_lines.append(f"  - {text}")
                        processed_elements.add(id(li))
                markdown_lines.append("")  # Add blank line after list
                processed_elements.add(id(element))
                
            elif element.name == 'pre':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(f"\n```\n{text}\n```\n")
                    processed_elements.add(id(element))
                    
            elif element.name == 'div':
                # Only process div if it has direct text content (not just children)
                direct_text = ''.join([str(s) for s in element.strings if s.parent == element]).strip()
                if direct_text and len(direct_text) > 20:
                    markdown_lines.append(f"{direct_text}\n")
                    processed_elements.add(id(element))
        
        # Join and clean up
        markdown = '\n'.join(markdown_lines)
        
        # Clean up excessive whitespace
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()
    
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
        
        # Extract title - try multiple strategies
        title = "Untitled"
        
        # Try h1 with specific class first
        h1_title = soup.find('h1', class_=re.compile(r'title|entry-title|post-title|article-title'))
        if h1_title:
            title = h1_title.get_text().strip()
        else:
            # Try any h1
            h1_elem = soup.find('h1')
            if h1_elem:
                title = h1_elem.get_text().strip()
            else:
                # Fall back to title tag
                title_elem = soup.find('title')
                if title_elem:
                    title = title_elem.get_text().strip()
                    # Clean up title (remove site name)
                    if '|' in title:
                        title = title.split('|')[0].strip()
                    elif '–' in title:
                        title = title.split('–')[0].strip()
                    elif '-' in title and len(title.split('-')) <= 3:
                        title = title.split('-')[0].strip()
        
        # First, try to find the main content container
        main_content = None
        
        # Strategy 1: Look for article or post content with specific classes
        content_selectors = [
            ('article', {'class': re.compile(r'post|entry|article|content', re.I)}),
            ('div', {'class': re.compile(r'post-content|entry-content|article-content|main-content|page-content|elementor.*post', re.I)}),
            ('div', {'id': re.compile(r'post|entry|article|content|main', re.I)}),
            ('main', {}),
            ('article', {}),
        ]
        
        for tag, attrs in content_selectors:
            main_content = soup.find(tag, attrs)
            if main_content:
                break
        
        # Strategy 2: If still not found, look for the largest div with substantial text
        if not main_content:
            all_divs = soup.find_all('div')
            if all_divs:
                # Filter divs that have enough text content (likely to be main content)
                content_divs = [d for d in all_divs if len(d.get_text(strip=True)) > 500]
                if content_divs:
                    main_content = max(content_divs, key=lambda d: len(d.get_text(strip=True)))
        
        # Strategy 3: Fall back to body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return title, ""
        
        # Now clean up unwanted elements ONLY from the main content
        for unwanted in main_content.find_all(['script', 'style', 'iframe', 'noscript']):
            unwanted.decompose()
        
        # Remove navigation elements that are inside the main content
        for unwanted in main_content.find_all(['nav']):
            unwanted.decompose()
        
        # Remove specific navigation/menu classes (be very specific)
        for unwanted in main_content.find_all(class_=re.compile(r'(^menu-|^nav-|comment-form|breadcrumb)', re.I)):
            unwanted.decompose()
        
        markdown_content = self._html_to_markdown(main_content)
        
        # Clean up excessive whitespace
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
        markdown_content = markdown_content.strip()
        
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
            # Check if this might be a JavaScript-rendered site
            print(f"⚠️  No content extracted from {url}")
            
            # Try to detect if it's a JS-heavy site
            html_to_check = response.text if 'response' in locals() else ""
            if html_to_check:
                has_elementor = 'elementor' in html_to_check.lower()
                has_react = 'react' in html_to_check.lower() or '__NEXT_DATA__' in html_to_check
                has_vue = 'vue' in html_to_check.lower() or 'data-v-' in html_to_check
                
                if has_react or has_vue or has_elementor:
                    print(f"💡 This site appears to use JavaScript frameworks (React/Vue/Elementor)")
                    print(f"💡 Try: 1) Save page manually in browser, 2) Look for RSS feed, 3) Use browser automation")
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

