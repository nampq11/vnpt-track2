#!/usr/bin/env bash
# Web crawler CLI for VNPT Track 2 project
# Usage: ./bin/crawl.sh [OPTIONS]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="data/data"
DELAY=1.0
FORCE=false
CATEGORY=""
MODE="url"
INPUT=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
${BLUE}VNPT Track 2 - Web Crawler${NC}

Usage: ./bin/crawl.sh [OPTIONS]

${YELLOW}Modes:${NC}
  -u, --url <url>              Crawl a single URL
  -f, --file <filepath>        Crawl URLs from a file (one per line)
  -l, --list <url1,url2,...>   Crawl multiple URLs (comma-separated)

${YELLOW}Options:${NC}
  -c, --category <name>        Category subfolder name
  -o, --output <dir>           Output directory (default: data/data)
  -d, --delay <seconds>        Delay between requests (default: 1.0)
  --force                      Overwrite existing files
  -h, --help                   Show this help message

${YELLOW}Examples:${NC}
  # Crawl a single Wikipedia page
  ./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/Việt_Nam" -c "Dia_ly_viet_nam"

  # Crawl multiple URLs
  ./bin/crawl.sh -l "https://vi.wikipedia.org/wiki/Hà_Nội,https://vi.wikipedia.org/wiki/Hồ_Chí_Minh" -c "Thanh_pho"

  # Crawl from a file
  ./bin/crawl.sh -f urls.txt -c "Lich_Su_Viet_nam" --delay 2.0

  # Force overwrite existing files
  ./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/Việt_Nam" --force

${YELLOW}URL File Format:${NC}
  Create a text file with one URL per line:
  
  https://vi.wikipedia.org/wiki/Việt_Nam
  https://vi.wikipedia.org/wiki/Hà_Nội
  # Lines starting with # are comments
  https://vi.wikipedia.org/wiki/Hồ_Chí_Minh

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            MODE="url"
            INPUT="$2"
            shift 2
            ;;
        -f|--file)
            MODE="file"
            INPUT="$2"
            shift 2
            ;;
        -l|--list)
            MODE="list"
            INPUT="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate input
if [[ -z "$INPUT" ]]; then
    print_error "No input provided. Use -u, -f, or -l to specify what to crawl."
    echo ""
    show_usage
    exit 1
fi

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Print configuration
print_info "Starting crawler with configuration:"
echo "  Mode:     $MODE"
echo "  Input:    $INPUT"
echo "  Output:   $OUTPUT_DIR"
echo "  Category: ${CATEGORY:-<auto-detect>}"
echo "  Delay:    ${DELAY}s"
echo "  Force:    $FORCE"
echo ""

# Check if input file exists (for file mode)
if [[ "$MODE" == "file" ]] && [[ ! -f "$INPUT" ]]; then
    print_error "File not found: $INPUT"
    exit 1
fi

# Run the crawler
print_info "Running crawler..."
echo ""

cd "$PROJECT_ROOT"

if uv run python -m src.utils.crawl_cli "$MODE" "$INPUT" "$OUTPUT_DIR" "$CATEGORY" "$DELAY" "$FORCE"; then
    echo ""
    print_success "Crawling completed successfully!"
    print_info "Files saved to: $OUTPUT_DIR/${CATEGORY}"
else
    echo ""
    print_error "Crawling failed. See errors above."
    exit 1
fi

print_success "Done!"

