#!/usr/bin/env python3
"""CLI entry point for web crawler."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.crawler import WebCrawler


async def main():
    """Main CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: crawl_cli.py <mode> <input> <output_dir> <category> <delay> <force>")
        sys.exit(1)
    
    mode = sys.argv[1]
    input_data = sys.argv[2]
    output_dir = sys.argv[3]
    category = sys.argv[4] if sys.argv[4] else None
    delay = float(sys.argv[5])
    force = sys.argv[6] == 'true'
    
    crawler = WebCrawler(output_dir=output_dir)
    
    try:
        if mode == 'url':
            await crawler.crawl_url(input_data, category=category, force=force)
        elif mode == 'file':
            await crawler.crawl_from_file(input_data, category=category, force=force, delay=delay)
        elif mode == 'list':
            urls = [u.strip() for u in input_data.split(',') if u.strip()]
            await crawler.crawl_urls(urls, category=category, force=force, delay=delay)
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
    finally:
        await crawler.close()


if __name__ == '__main__':
    asyncio.run(main())

