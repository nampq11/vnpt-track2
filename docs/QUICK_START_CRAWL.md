# 🚀 Quick Start: Crawl & Add Data

**3-Step Process to add data to existing domains**

---

## ⚡ TL;DR

```bash
# 1. Crawl data
./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/YOUR_TOPIC" \
  -c "category_name"

# 2. Update index
./bin/knowledge.sh upsert \
  --data-dir data/data/category_name \
  --provider azure

# 3. Verify
./bin/knowledge.sh info
```

---

## 📋 Category Quick Reference

### POLITICS Domain
```bash
# Government
-c "cong_hoa_xa_hoi_chu_nghia_VN"

# Party
-c "dang_cong_san_viet_nam"

# Political figures
-c "nhan_vat_chinh_tri"

# Defense
-c "Quoc_phong_Viet_nam"
```

### LAW Domain
```bash
-c "Phap_luat_Viet_Nam"    # General law
-c "hien_phap"              # Constitution
-c "Quyen_nghia_vu"         # Rights & duties
```

### HISTORY Domain
```bash
-c "Lich_Su_Viet_nam"       # General history
-c "Bac_Ho"                 # Ho Chi Minh
-c "khang_chien_lon"        # Major wars
```

### GEOGRAPHY Domain
```bash
-c "Dia_ly_viet_nam"        # General geography
-c "Dia_chinh_Viet_nam"     # Administrative
-c "Dia_dien_du_lich"       # Tourism
```

### CULTURE Domain
```bash
-c "Van_Hoa_Viet_Nam"       # General culture
-c "Van_hoa_am_thuc"        # Cuisine
-c "le_hoi_truyen_thong"    # Festivals
```

---

## 📝 Common Examples

### Example 1: Add Single URL

```bash
./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/Quốc_hội_Việt_Nam" \
  -c "cong_hoa_xa_hoi_chu_nghia_VN"

./bin/knowledge.sh upsert \
  --data-dir data/data/cong_hoa_xa_hoi_chu_nghia_VN \
  --provider azure
```

### Example 2: Add Multiple URLs

```bash
./bin/crawl.sh -l "\
https://vi.wikipedia.org/wiki/Chính_phủ_Việt_Nam,\
https://vi.wikipedia.org/wiki/Quốc_hội_Việt_Nam\
" -c "cong_hoa_xa_hoi_chu_nghia_VN" --delay 2.0

./bin/knowledge.sh upsert \
  --data-dir data/data/cong_hoa_xa_hoi_chu_nghia_VN \
  --provider azure
```

### Example 3: Add from File

```bash
# Create URL list
cat > urls.txt << EOF
https://vi.wikipedia.org/wiki/Đại_hội_Đảng_XIII
https://vi.wikipedia.org/wiki/Tổng_Bí_thư
EOF

# Crawl
./bin/crawl.sh -f urls.txt -c "dang_cong_san_viet_nam" --delay 2.0

# Index
./bin/knowledge.sh upsert \
  --data-dir data/data/dang_cong_san_viet_nam \
  --provider azure
```

---

## 🔧 Troubleshooting

### Crawler Issues
```bash
# Check URL accessibility
curl -I "https://vi.wikipedia.org/wiki/Việt_Nam"

# Increase delay if rate-limited
./bin/crawl.sh -u "..." --delay 3.0
```

### Index Issues
```bash
# Check index status
./bin/knowledge.sh info

# Rebuild if needed (slow!)
./bin/knowledge.sh build --data-dir data/data --provider azure
```

### Category Not Found
```bash
# List existing categories
ls data/data/

# Create new category
mkdir -p data/data/your_category
```

---

## 📚 Full Documentation

- Complete Guide: `docs/CRAWL_AND_INDEX_GUIDE.md`
- Working Example: `examples/add_politics_data_example.sh`
- Domain Mapper: `src/brain/agent/domain_mapper.py`

---

## 🎯 Cheat Sheet

| Action | Command |
|--------|---------|
| Crawl single URL | `./bin/crawl.sh -u "URL" -c "category"` |
| Crawl from file | `./bin/crawl.sh -f urls.txt -c "category"` |
| Update index | `./bin/knowledge.sh upsert --data-dir data/data/category --provider azure` |
| Check status | `./bin/knowledge.sh info` |
| List categories | `ls data/data/` |
| Delete category | `./bin/knowledge.sh delete --category "name"` |
| Rebuild index | `./bin/knowledge.sh build --data-dir data/data --provider azure` |

---

**Need More Help?**
- Crawler: `./bin/crawl.sh --help`
- Knowledge Manager: `./bin/knowledge.sh --help`
- Full Guide: `docs/CRAWL_AND_INDEX_GUIDE.md`

