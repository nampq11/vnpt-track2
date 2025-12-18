# 📚 Guide: Crawl & Add Data to Existing Domains

Complete workflow for crawling new data and integrating it into the RAG knowledge base.

---

## 🎯 Overview

**3-Step Process**:
1. **Crawl** new data from web sources
2. **Organize** into domain-specific categories
3. **Index** to update the vector database

---

## 📥 Step 1: Crawl New Data

### Option A: Crawl Single URL

```bash
./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/Quốc_hội_Việt_Nam" \
  -c "cong_hoa_xa_hoi_chu_nghia_VN"
```

**Parameters**:
- `-u <url>`: Wikipedia URL to crawl
- `-c <category>`: Target category (matches folder name in `data/data/`)

### Option B: Crawl Multiple URLs

```bash
./bin/crawl.sh -l "URL1,URL2,URL3" \
  -c "dang_cong_san_viet_nam" \
  --delay 2.0
```

**Example - Add to POLITICS domain**:
```bash
./bin/crawl.sh -l "\
https://vi.wikipedia.org/wiki/Chính_phủ_Việt_Nam,\
https://vi.wikipedia.org/wiki/Quốc_hội_Việt_Nam,\
https://vi.wikipedia.org/wiki/Thủ_tướng_Chính_phủ_Việt_Nam\
" -c "cong_hoa_xa_hoi_chu_nghia_VN" --delay 2.0
```

### Option C: Crawl from URL File

**1. Create URL list** (`urls.txt`):
```txt
# POLITICS domain - Government structure
https://vi.wikipedia.org/wiki/Chính_phủ_Việt_Nam
https://vi.wikipedia.org/wiki/Quốc_hội_Việt_Nam
https://vi.wikipedia.org/wiki/Tòa_án_nhân_dân_Việt_Nam

# Party documents
https://vi.wikipedia.org/wiki/Đại_hội_Đảng_toàn_quốc_lần_thứ_XIII
```

**2. Run crawler**:
```bash
./bin/crawl.sh -f urls.txt \
  -c "dang_cong_san_viet_nam" \
  --delay 2.0
```

### Crawler Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `-u, --url` | Single URL | - | `-u "https://..."` |
| `-f, --file` | URL list file | - | `-f urls.txt` |
| `-l, --list` | Comma-separated URLs | - | `-l "url1,url2"` |
| `-c, --category` | Category folder | auto-detect | `-c "Lich_Su_Viet_nam"` |
| `-o, --output` | Output directory | `data/data` | `-o data/new` |
| `-d, --delay` | Delay between requests (sec) | 1.0 | `--delay 2.0` |
| `--force` | Overwrite existing files | false | `--force` |

---

## 📂 Step 2: Organize Into Domain Categories

### 2.1 Choose the Right Category

Each **domain** maps to multiple **categories**. Choose based on your content:

#### **POLITICS Domain** (6 categories):
```
cong_hoa_xa_hoi_chu_nghia_VN    # Government structure, state institutions
dang_cong_san_viet_nam          # Party organization, congresses
nhan_vat_chinh_tri              # Political figures, leaders
Quoc_phong_Viet_nam             # National defense, military
Bac_Ho                          # Ho Chi Minh (as political figure)
Tu_Tuong_HCM                    # Ho Chi Minh ideology
```

#### **LAW Domain** (5 categories):
```
Phap_luat_Viet_Nam             # General legal system
hien_phap                      # Constitution
Quyen_nghia_vu                 # Rights and obligations
cong_hoa_xa_hoi_chu_nghia_VN   # State law framework
dang_cong_san_viet_nam         # Party regulations
```

#### **HISTORY Domain** (7 categories):
```
Lich_Su_Viet_nam               # General history
Bac_Ho                         # Ho Chi Minh (historical)
khang_chien_lon                # Major resistance wars
nhan_vat_lich_su_tieu_bieu     # Historical figures
nhan_vat_chinh_tri             # Political history
Tu_Tuong_HCM                   # Historical ideology
dang_cong_san_viet_nam         # Party history
```

#### **GEOGRAPHY Domain** (6 categories):
```
Dia_ly_viet_nam                # General geography
Dia_chinh_Viet_nam             # Administrative geography
Dia_hinh_Viet_Nam              # Terrain, topography
Dia_dien_du_lich               # Tourism destinations
Khi_hau_thoi_tiet_thien_tai    # Climate, weather
song_ngoi_bien_tai_nguyen_thien_nhien  # Natural resources
```

#### **CULTURE Domain** (7 categories):
```
Van_Hoa_Viet_Nam               # General culture
Van_hoa_am_thuc                # Cuisine
van_hoa_lang_xa                # Village culture
Van_hoa_ung_xu                 # Behavior, etiquette
Phong_tuc_tap_quan             # Customs, traditions
le_hoi_truyen_thong            # Festivals
tin_nguong_ton_giao            # Religion, beliefs
```

### 2.2 Manual File Organization

If you want to manually place files:

```bash
# Check category structure
ls data/data/

# Move crawled file to specific category
cp my_document.txt data/data/cong_hoa_xa_hoi_chu_nghia_VN/

# Or create new subcategory (will be auto-detected)
mkdir -p data/data/my_new_politics_category
cp *.txt data/data/my_new_politics_category/
```

### 2.3 Verify File Placement

```bash
# Check files in category
ls data/data/cong_hoa_xa_hoi_chu_nghia_VN/

# Count files
find data/data/dang_cong_san_viet_nam -name "*.txt" | wc -l
```

---

## 🔄 Step 3: Update Knowledge Index

### 3.1 Check Current Index Status

```bash
./bin/knowledge.sh info
```

**Output**:
```
📊 Knowledge Base Status
  Location: data/embeddings/knowledge/knowledge.lance/
  Vectors: 26,513 chunks
  Dimension: 1536 (Azure)
  Categories: 26
```

### 3.2 Option A: Incremental Update (Recommended)

**Fast**: Add only new documents (~30 sec per 100 docs)

```bash
# Add new category or updated files
./bin/knowledge.sh upsert \
  --data-dir data/data/dang_cong_san_viet_nam \
  --provider azure
```

**Parameters**:
- `--data-dir`: Directory with new/updated files
- `--provider`: Embedding provider (`azure` or `vnpt`)
- `--chunk-size`: Text chunk size (default: 512)
- `--overlap`: Chunk overlap (default: 50)

**Example - Add to multiple categories**:
```bash
# Upsert POLITICS domain categories
./bin/knowledge.sh upsert \
  --data-dir data/data/cong_hoa_xa_hoi_chu_nghia_VN \
  --provider azure

./bin/knowledge.sh upsert \
  --data-dir data/data/dang_cong_san_viet_nam \
  --provider azure

./bin/knowledge.sh upsert \
  --data-dir data/data/nhan_vat_chinh_tri \
  --provider azure
```

### 3.3 Option B: Full Rebuild

**Slow**: Rebuild entire index (~10 minutes)

```bash
./bin/knowledge.sh build \
  --data-dir data/data \
  --provider azure
```

**When to use**:
- Major restructuring of categories
- Changing embedding provider
- Index corruption
- First-time setup

### 3.4 Delete Outdated Content

**Delete by file**:
```bash
./bin/knowledge.sh delete \
  --file "data/data/old_category/outdated_doc.txt"
```

**Delete entire category**:
```bash
./bin/knowledge.sh delete \
  --category "old_category_name"
```

### 3.5 Verify Index Update

```bash
# Check updated stats
./bin/knowledge.sh info

# Should show increased vector count
```

---

## 🎯 Complete Example Workflows

### Example 1: Add POLITICS Data (Government Documents)

**Goal**: Add information about Vietnamese government structure

```bash
# Step 1: Create URL list
cat > politics_urls.txt << EOF
https://vi.wikipedia.org/wiki/Chính_phủ_Việt_Nam
https://vi.wikipedia.org/wiki/Quốc_hội_Việt_Nam
https://vi.wikipedia.org/wiki/Tổng_thống_Việt_Nam
https://vi.wikipedia.org/wiki/Thủ_tướng_Chính_phủ_Việt_Nam
EOF

# Step 2: Crawl URLs
./bin/crawl.sh -f politics_urls.txt \
  -c "cong_hoa_xa_hoi_chu_nghia_VN" \
  --delay 2.0

# Step 3: Verify files
ls data/data/cong_hoa_xa_hoi_chu_nghia_VN/

# Step 4: Update index
./bin/knowledge.sh upsert \
  --data-dir data/data/cong_hoa_xa_hoi_chu_nghia_VN \
  --provider azure

# Step 5: Verify
./bin/knowledge.sh info
```

### Example 2: Add POLITICS Data (Party History)

```bash
# Step 1: Crawl party documents
./bin/crawl.sh -l "\
https://vi.wikipedia.org/wiki/Đại_hội_Đảng_Cộng_sản_Việt_Nam_lần_thứ_XIII,\
https://vi.wikipedia.org/wiki/Ban_Chấp_hành_Trung_ương_Đảng_Cộng_sản_Việt_Nam,\
https://vi.wikipedia.org/wiki/Tổng_Bí_thư_Đảng_Cộng_sản_Việt_Nam\
" -c "dang_cong_san_viet_nam" --delay 2.0

# Step 2: Update index
./bin/knowledge.sh upsert \
  --data-dir data/data/dang_cong_san_viet_nam \
  --provider azure
```

### Example 3: Add New Category to Existing Domain

```bash
# Step 1: Create new category folder
mkdir -p data/data/ngoai_giao_viet_nam

# Step 2: Crawl diplomatic content
./bin/crawl.sh -f diplomatic_urls.txt \
  -c "ngoai_giao_viet_nam" \
  --delay 2.0

# Step 3: Add category to domain mapper
# Edit: src/brain/agent/domain_mapper.py
# Add "ngoai_giao_viet_nam" to POLITICS or LAW domain

# Step 4: Index new category
./bin/knowledge.sh upsert \
  --data-dir data/data/ngoai_giao_viet_nam \
  --provider azure

# Step 5: Run tests
uv run pytest tests/test_domain_aware_rag.py -v
```

---

## 📋 Best Practices

### ✅ DO:
- **Use descriptive categories**: Match existing naming conventions
- **Set appropriate delays**: `--delay 2.0` for Wikipedia (respect rate limits)
- **Verify before indexing**: Check crawled files look correct
- **Incremental updates**: Use `upsert` for new data
- **Backup before rebuild**: Copy `data/embeddings/knowledge/` before full rebuild
- **Test after changes**: Run domain mapper tests

### ❌ DON'T:
- **Don't crawl aggressively**: Respect website rate limits
- **Don't mix languages**: Keep Vietnamese content separate
- **Don't duplicate categories**: Use existing categories when possible
- **Don't skip verification**: Always check index info after updates
- **Don't force rebuild**: Unless necessary (slow process)

---

## 🔧 Troubleshooting

### Issue: Crawler Fails

**Check**:
```bash
# Verify URL is accessible
curl -I "https://vi.wikipedia.org/wiki/Việt_Nam"

# Check network connection
ping vi.wikipedia.org

# Increase delay
./bin/crawl.sh -u "..." --delay 3.0
```

### Issue: Category Not Recognized

**Solution**: Verify category exists
```bash
# List existing categories
ls data/data/

# Create category if needed
mkdir -p data/data/your_category_name
```

### Issue: Index Update Fails

**Check**:
```bash
# Verify provider configuration
cat config/vnpt.json  # For VNPT
# or check Azure credentials in .env

# Check disk space
df -h data/embeddings/

# Rebuild if corrupted
./bin/knowledge.sh build --data-dir data/data --provider azure
```

### Issue: Files Not Being Indexed

**Verify**:
```bash
# Check file format (.txt required)
file data/data/category/*.txt

# Check file permissions
ls -la data/data/category/

# Check file encoding (UTF-8 required)
file -i data/data/category/*.txt
```

---

## 📊 Monitoring & Verification

### Check Index Health

```bash
# Basic info
./bin/knowledge.sh info

# Detailed log
./bin/knowledge.sh info --verbose

# Check specific category count
find data/data/dang_cong_san_viet_nam -name "*.txt" | wc -l
```

### Test Retrieval

```python
# Quick test script
from src.brain.rag.lancedb_retriever import LanceDBRetriever
from src.brain.llm.services.azure import AzureService

llm = AzureService()
retriever = LanceDBRetriever.from_directory(
    "data/embeddings/knowledge",
    llm_service=llm
)

# Test query
results = await retriever.retrieve(
    query="Quốc hội Việt Nam có bao nhiêu đại biểu?",
    categories_filter=["cong_hoa_xa_hoi_chu_nghia_VN"],
    top_k=3
)

for r in results:
    print(f"Score: {r.score:.2f} | {r.content[:100]}...")
```

---

## 🚀 Quick Reference

### Common Commands

```bash
# Crawl single URL to POLITICS domain
./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/..." \
  -c "cong_hoa_xa_hoi_chu_nghia_VN"

# Crawl from file
./bin/crawl.sh -f urls.txt -c "dang_cong_san_viet_nam" --delay 2.0

# Update index (incremental)
./bin/knowledge.sh upsert --data-dir data/data/category --provider azure

# Rebuild index (full)
./bin/knowledge.sh build --data-dir data/data --provider azure

# Check status
./bin/knowledge.sh info

# Delete category
./bin/knowledge.sh delete --category "old_category"
```

### Domain-Category Quick Reference

```
POLITICS:  cong_hoa_xa_hoi_chu_nghia_VN, dang_cong_san_viet_nam, 
           nhan_vat_chinh_tri, Quoc_phong_Viet_nam, Bac_Ho, Tu_Tuong_HCM

LAW:       Phap_luat_Viet_Nam, hien_phap, Quyen_nghia_vu

HISTORY:   Lich_Su_Viet_nam, Bac_Ho, khang_chien_lon, nhan_vat_lich_su_tieu_bieu

GEOGRAPHY: Dia_ly_viet_nam, Dia_chinh_Viet_nam, Dia_hinh_Viet_Nam, Dia_dien_du_lich

CULTURE:   Van_Hoa_Viet_Nam, Van_hoa_am_thuc, le_hoi_truyen_thong, Phong_tuc_tap_quan
```

---

**Need Help?**
- Crawler Help: `./bin/crawl.sh --help`
- Knowledge Manager Help: `./bin/knowledge.sh --help`
- Test Crawler: `uv run pytest tests/test_crawler.py -v`
- Test Indexing: `uv run pytest tests/test_rag_indexing.py -v`

