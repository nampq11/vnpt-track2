#!/usr/bin/env python3
"""Generate sample artifacts for testing runtime pipeline

This script creates minimal but functional indices for testing:
- FAISS vector index
- BM25 keyword index  
- Safety vector matrix
- Metadata JSON

Usage:
    uv run python scripts/generate_sample_artifacts.py
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import Chunk, ChunkMetadata, ChunkType
from src.core.config import Config
from src.offline.indexer.faiss_builder import FAISSIndexBuilder
from src.offline.indexer.bm25_builder import BM25IndexBuilder
from src.offline.indexer.safety_builder import SafetyIndexBuilder
from src.brain.llm.services.ollama import OllamaService


async def main():
    print("=" * 70)
    print("GENERATING SAMPLE ARTIFACTS FOR RUNTIME TESTING")
    print("=" * 70)
    
    # Initialize
    config = Config.from_env()
    output_dir = Path(config.offline.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create knowledge base
    kb_file = Path("data/knowledge_base.json")
    if kb_file.exists():
        print(f"\n✓ Loading knowledge base from {kb_file}")
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb_json = json.load(f)
            
        # Handle different formats
        if isinstance(kb_json, dict) and "documents" in kb_json:
            # New format: {"documents": ["text1", "text2", ...]}
            print("   Format: JSON with 'documents' array (string format)")
            knowledge_data = kb_json["documents"]
        elif isinstance(kb_json, list):
            # Legacy format: [{"title": "...", "content": "...", ...}, ...]
            print("   Format: Array of objects")
            knowledge_data = kb_json
        else:
            print("   ⚠️ Unknown format, using sample data")
            knowledge_data = create_sample_knowledge()
    else:
        print(f"\n⚠️  {kb_file} not found, creating sample data...")
        knowledge_data = create_sample_knowledge()
        # Save for future use
        kb_file.parent.mkdir(parents=True, exist_ok=True)
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved sample knowledge base to {kb_file}")
    
    # Create chunks
    print(f"\n📦 Step 1: Creating chunks from {len(knowledge_data)} documents...")
    chunks = create_chunks_from_knowledge(knowledge_data)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Initialize LLM service for embeddings
    print("\n🤖 Step 2: Initializing LLM service for embeddings...")
    try:
        llm_service = OllamaService(model=config.ollama.model)
        print(f"   ✓ Using Ollama model: {config.ollama.model}")
    except Exception as e:
        print(f"   ✗ Failed to initialize Ollama: {str(e)}")
        print("\n💡 Make sure Ollama is running:")
        print("   curl http://localhost:11434/v1/models")
        return 1
    
    # Build FAISS index
    print("\n🔍 Step 3: Building FAISS vector index...")
    try:
        faiss_builder = FAISSIndexBuilder(llm_service)
        print("   ⏳ Generating embeddings (this may take a minute)...")
        faiss_index, embeddings = await faiss_builder.build(chunks, embedding_dim=-1)
        
        faiss_path = output_dir / "faiss.index"
        faiss_builder.save(str(faiss_path))
        print(f"   ✓ Saved FAISS index to {faiss_path}")
        print(f"   ✓ Index size: {len(embeddings)} vectors × {embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"   ✗ FAISS build failed: {str(e)}")
        return 1
    
    # Build BM25 index
    print("\n🔤 Step 4: Building BM25 keyword index...")
    try:
        bm25_builder = BM25IndexBuilder()
        bm25_index = bm25_builder.build(chunks)
        
        bm25_path = output_dir / "bm25.pkl"
        bm25_builder.save(str(bm25_path))
        print(f"   ✓ Saved BM25 index to {bm25_path}")
    except Exception as e:
        print(f"   ✗ BM25 build failed: {str(e)}")
        return 1
    
    # Build Safety index
    print("\n🛡️  Step 5: Building Safety vector index...")
    try:
        safety_builder = SafetyIndexBuilder(llm_service)
        harmful_questions = safety_builder.generate_synthetic_questions()
        print(f"   ⏳ Generating {len(harmful_questions)} safety vectors...")
        
        safety_vectors = await safety_builder.build(harmful_questions)
        
        safety_path = output_dir / "safety.npy"
        safety_builder.save(str(safety_path))
        print(f"   ✓ Saved safety vectors to {safety_path}")
        print(f"   ✓ Generated {len(harmful_questions)} harmful question vectors")
    except Exception as e:
        print(f"   ✗ Safety build failed: {str(e)}")
        return 1
    
    # Save metadata
    print("\n📄 Step 6: Saving chunk metadata...")
    try:
        metadata_path = output_dir / "metadata.json"
        save_metadata(chunks, metadata_path)
        print(f"   ✓ Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"   ✗ Metadata save failed: {str(e)}")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ARTIFACTS GENERATION COMPLETE")
    print("=" * 70)
    print(f"📊 Total chunks: {len(chunks)}")
    print(f"🔍 FAISS vectors: {len(embeddings)}")
    print(f"🛡️  Safety vectors: {len(harmful_questions)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n✨ You can now run runtime tests:")
    print(f"   uv run pytest tests/test_runtime_rag.py -v")
    print(f"   uv run python predict.py --mode test --input data/test.json")
    
    return 0


def create_sample_knowledge():
    """Create sample Vietnamese knowledge base"""
    return [
        {
            "title": "Luật Đất đai 2024",
            "content": """Luật Đất đai năm 2024 được Quốc hội thông qua ngày 18 tháng 1 năm 2024, 
            có hiệu lực từ ngày 1 tháng 1 năm 2025. Luật này quy định về chế độ sở hữu, 
            quyền sử dụng đất, nghĩa vụ và trách nhiệm của người sử dụng đất. 
            
            Điều 4: Nhà nước thống nhất quản lý về đất đai trong phạm vi cả nước.
            
            Điều 10: Người sử dụng đất có quyền chuyển đổi, chuyển nhượng, cho thuê, 
            cho thuê lại, thừa kế, tặng cho quyền sử dụng đất theo quy định của Luật này.""",
            "year": 2024,
            "type": "LAW",
            "province": "ALL"
        },
        {
            "title": "Hiến pháp 2013",
            "content": """Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam năm 2013 
            được Quốc hội khóa XIII, kỳ họp thứ 6 thông qua ngày 28 tháng 11 năm 2013.
            
            Điều 1: Nước Cộng hòa xã hội chủ nghĩa Việt Nam là một nước độc lập, 
            có chủ quyền, thống nhất và toàn vẹn lãnh thổ, bao gồm đất liền, hải đảo, 
            vùng biển và vùng trời.
            
            Điều 2: Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam là Nhà nước pháp quyền 
            xã hội chủ nghĩa của nhân dân, do nhân dân, vì nhân dân.""",
            "year": 2013,
            "type": "LAW",
            "province": "ALL"
        },
        {
            "title": "Toán học - Đạo hàm cơ bản",
            "content": """Đạo hàm là một trong những khái niệm cơ bản trong giải tích.
            
            Định nghĩa: Đạo hàm của hàm số f(x) tại điểm x₀ là giới hạn:
            f'(x₀) = lim[h→0] (f(x₀+h) - f(x₀))/h
            
            Các công thức đạo hàm cơ bản:
            - (c)' = 0 (c là hằng số)
            - (x^n)' = n·x^(n-1)
            - (sin x)' = cos x
            - (cos x)' = -sin x
            - (e^x)' = e^x
            - (ln x)' = 1/x
            
            Quy tắc tổng: (f + g)' = f' + g'
            Quy tắc tích: (f·g)' = f'·g + f·g'
            Quy tắc thương: (f/g)' = (f'·g - f·g')/g²""",
            "year": 2020,
            "type": "MATH",
            "province": "ALL"
        },
        {
            "title": "Vật lý - Điện trở và định luật Ohm",
            "content": """Điện trở là đại lượng đặc trưng cho mức độ cản trở dòng điện 
            của vật dẫn.
            
            Công thức tính điện trở: R = ρ·L/S
            Trong đó:
            - R: điện trở (Ω)
            - ρ: điện trở suất của vật liệu (Ω·m)
            - L: chiều dài dây dẫn (m)
            - S: tiết diện dây dẫn (m²)
            
            Định luật Ohm: U = I·R
            Trong đó:
            - U: hiệu điện thế (V)
            - I: cường độ dòng điện (A)
            - R: điện trở (Ω)
            
            Công suất điện: P = U·I = I²·R = U²/R""",
            "year": 2020,
            "type": "MATH",
            "province": "ALL"
        },
        {
            "title": "Lịch sử - Cách mạng tháng Tám 1945",
            "content": """Cách mạng tháng Tám năm 1945 là cuộc cách mạng giải phóng dân tộc 
            của nhân dân Việt Nam do Đảng Cộng sản Đông Dương và Chủ tịch Hồ Chí Minh lãnh đạo.
            
            Bối cảnh: Sau khi phát xít Nhật đầu hàng Đồng minh (15/8/1945), 
            tạo ra chân không quyền lực tại Việt Nam.
            
            Diễn biến:
            - 16/8/1945: Đại hội quốc dân ở Tân Trào quyết định tổng khởi nghĩa
            - 19/8/1945: Khởi nghĩa giành chính quyền ở Hà Nội
            - 23/8/1945: Cách mạng thành công tại Huế
            - 25/8/1945: Chính quyền cách mạng nắm Sài Gòn
            - 2/9/1945: Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập, 
              tuyên bố thành lập nước Việt Nam Dân chủ Cộng hòa
            
            Ý nghĩa: Lần đầu tiên trong lịch sử, nhân dân ta giành được chính quyền 
            trên phạm vi cả nước.""",
            "year": 1945,
            "type": "HISTORY",
            "province": "ALL"
        },
        {
            "title": "Kinh tế - GDP và tăng trưởng kinh tế",
            "content": """GDP (Gross Domestic Product - Tổng sản phẩm quốc nội) là 
            tổng giá trị thị trường của tất cả hàng hóa và dịch vụ cuối cùng được sản xuất 
            trong một quốc gia trong một khoảng thời gian nhất định (thường là một năm).
            
            Các phương pháp tính GDP:
            1. Phương pháp sản xuất: GDP = Σ Giá trị gia tăng
            2. Phương pháp thu nhập: GDP = Lương + Lợi nhuận + Thuế
            3. Phương pháp chi tiêu: GDP = C + I + G + (X - M)
               - C: Tiêu dùng cá nhân
               - I: Đầu tư
               - G: Chi tiêu chính phủ
               - X: Xuất khẩu
               - M: Nhập khẩu
            
            Tăng trưởng GDP = (GDP năm nay - GDP năm trước) / GDP năm trước × 100%
            
            GDP bình quân đầu người = GDP / Dân số""",
            "year": 2020,
            "type": "GENERAL",
            "province": "ALL"
        },
    ]


def create_chunks_from_knowledge(knowledge_data):
    """Convert knowledge base entries to Chunk objects
    
    Supports two formats:
    1. List of objects: [{"title": "...", "content": "...", "year": ..., "type": "..."}]
    2. List of strings: ["text1", "text2", ...]
    """
    chunks = []
    
    for idx, doc in enumerate(knowledge_data):
        chunk_id = f"chunk_{idx:05d}"
        
        # Handle different document formats
        if isinstance(doc, dict):
            # Format 1: Object with title, content, year, type
            text = doc.get("content", "").strip()
            source = doc.get("title", f"doc_{idx}")
            doc_type = doc.get("type", "GENERAL")
            year = doc.get("year", 2020)
            province = doc.get("province", "ALL")
        elif isinstance(doc, str):
            # Format 2: Plain string
            text = doc.strip()
            source = f"doc_{idx}"
            doc_type = "GENERAL"
            year = 2020
            province = "ALL"
        else:
            print(f"   ⚠️ Skipping unknown doc format at index {idx}")
            continue
        
        # Ensure type is valid
        try:
            chunk_type = ChunkType[doc_type]
        except KeyError:
            chunk_type = ChunkType.GENERAL
        
        # Create metadata
        metadata = ChunkMetadata(
            source=source,
            type=chunk_type,
            valid_from=year,
            expire_at=9999,
            province=province
        )
        
        # Create chunk
        chunk = Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata
        )
        
        chunks.append(chunk)
    
    return chunks


def save_metadata(chunks, output_path):
    """Save chunk metadata to JSON for reference"""
    metadata_list = []
    
    for chunk in chunks:
        metadata_list.append({
            "id": chunk.id,
            "text_preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
            "text_length": len(chunk.text),
            "source": chunk.metadata.source,
            "type": chunk.metadata.type.value,
            "valid_from": chunk.metadata.valid_from,
            "expire_at": chunk.metadata.expire_at,
            "province": chunk.metadata.province,
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

