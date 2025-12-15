"""Runtime inference pipeline - Main orchestrator for Docker"""

import asyncio
import json
from typing import List, Optional
from pathlib import Path

from src.core.models import Question, PredictionResult
from src.core.config import Config
from src.runtime.llm.base import LLMService
from src.runtime.router.regex_router import RegexRouter
from src.runtime.safety.guard import SafetyGuard
from src.runtime.safety.selector import SafetySelector
from src.runtime.rag.hybrid_search import HybridSearchEngine
from src.runtime.rag.temporal_filter import TemporalFilter
from src.runtime.prompts.stem_prompt import STEMPromptBuilder
from src.runtime.prompts.rag_prompt import RAGPromptBuilder
from src.runtime.prompts.reading_prompt import ReadingPromptBuilder


class RuntimeInferencePipeline:
    """Main runtime inference pipeline for Docker"""
    
    def __init__(
        self,
        llm_service: LLMService,
        config: Optional[Config] = None,
    ):
        """
        Initialize runtime pipeline
        
        Args:
            llm_service: LLM service (VNPT API in Docker)
            config: Configuration object
        """
        self.llm_service = llm_service
        self.config = config or Config()
        
        # Initialize components
        self.router = RegexRouter()
        self.safety_guard = SafetyGuard(llm_service)
        self.safety_selector = SafetySelector(llm_service)
        self.rag_engine = HybridSearchEngine(llm_service)
        self.temporal_filter = TemporalFilter()
        
        # Load artifacts if available
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load pre-built indices and safety vectors"""
        try:
            # Load safety vectors
            safety_path = self.config.runtime.artifacts_dir + "/safety.npy"
            if Path(safety_path).exists():
                self.safety_guard.load_from_file(safety_path)
                print(f"Loaded safety vectors from {safety_path}")
        except Exception as e:
            print(f"Warning: Could not load artifacts: {str(e)}")
    
    async def process_question(
        self,
        question: Question,
        use_rag: bool = True,
    ) -> PredictionResult:
        """
        Process single question through full pipeline
        
        Pipeline:
        1. Safety Check (semantic firewall)
        2. If unsafe: Safety Selector
        3. If safe: Router (READING/STEM/RAG)
        4. Generate prompt based on route
        5. Call LLM
        6. Parse answer
        
        Args:
            question: Question to process
            use_rag: Whether to enable RAG (if artifacts available)
            
        Returns:
            PredictionResult with answer
        """
        # Step 1: Safety Check
        safety_result = await self.safety_guard.check(question.question)
        
        if not safety_result.is_safe:
            # Step 2: Safety Selector
            answer = await self.safety_selector.select_answer(question)
            return PredictionResult(
                qid=question.qid,
                predicted_answer=answer,
                route_mode="SAFETY",
            )
        
        # Step 3: Route question
        route = self.router.route_with_context(question)
        
        # Step 4 & 5: Generate prompt and call LLM
        system_prompt = self._build_system_prompt(route.mode)
        
        if route.mode == "READING":
            # Reading comprehension - use passage as-is
            user_prompt = ReadingPromptBuilder.build(question)
        
        elif route.mode == "STEM":
            # Math/Science with CoT
            user_prompt = STEMPromptBuilder.build(question)
        
        else:  # RAG
            # Retrieve context and build prompt
            if use_rag:
                # Extract year from question for temporal filtering
                query_year = self.temporal_filter.extract_year(question.question)
                
                # Perform hybrid search
                search_results = await self.rag_engine.search(
                    question.question,
                    temporal_year=query_year,
                )
                
                route.context_chunks = [r.chunk for r in search_results.results]
                route.extracted_year = query_year
            
            # Build RAG prompt
            if route.context_chunks:
                user_prompt = RAGPromptBuilder.build(
                    question,
                    [r for r in (await self._get_search_results(route)) or []],
                )
            else:
                # Fallback to simple prompt without context
                user_prompt = self._build_simple_prompt(question)
        
        # Call LLM
        try:
            response = await self.llm_service.generate(
                user_input=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=256,
            )
            
            # Parse answer
            answer = self._extract_answer(response.content)
        except Exception as e:
            print(f"LLM error for {question.qid}: {str(e)}")
            answer = "A"  # Default fallback
        
        return PredictionResult(
            qid=question.qid,
            predicted_answer=answer,
            route_mode=route.mode,
        )
    
    async def _get_search_results(self, route):
        """
        Convert route chunks to SearchResult objects for RAG prompting
        
        Args:
            route: QueryRoute with context_chunks
            
        Returns:
            List of SearchResult objects or empty list
        """
        if not hasattr(route, 'context_chunks') or not route.context_chunks:
            return []
        
        from src.core.models import SearchResult
        
        # Convert chunks to SearchResult format with mock scores
        # In a full implementation, these scores would come from the hybrid search
        results = []
        for i, chunk in enumerate(route.context_chunks):
            # Assign decreasing scores for ranking (higher rank = lower score)
            rank = i + 1
            score = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, etc.
            
            results.append(SearchResult(
                chunk=chunk,
                keyword_score=max(score, 0.1),  # Minimum 0.1
                semantic_score=max(score, 0.1),
                final_score=max(score, 0.1),
                rank=rank
            ))
        
        return results
    
    def _build_system_prompt(self, route_mode: str) -> str:
        """Build system prompt based on route mode"""
        if route_mode == "STEM":
            return """Bạn là một chuyên gia toán học và khoa học. 
Hãy giải quyết các bài toán bằng cách suy nghĩ từng bước và cung cấp lý do rõ ràng.
Cuối cùng, hãy đưa ra đáp án dưới dạng một chữ cái: A, B, C hoặc D."""
        
        elif route_mode == "READING":
            return """Bạn là một chuyên gia phân tích văn bản.
Hãy đọc kỹ đoạn văn bản được cung cấp và trả lời các câu hỏi dựa hoàn toàn trên thông tin trong văn bản.
Đừng sử dụng kiến thức bên ngoài.
Cuối cùng, hãy đưa ra đáp án dưới dạng một chữ cái: A, B, C hoặc D."""
        
        else:  # RAG
            return """Bạn là một trợ lý thông minh.
Hãy trả lời câu hỏi dựa trên thông tin được cung cấp trong ngữ cảnh.
Nếu thông tin không có, hãy nêu rõ điều đó.
Cuối cùng, hãy đưa ra đáp án dưới dạng một chữ cái: A, B, C hoặc D."""
    
    def _build_simple_prompt(self, question: Question) -> str:
        """Build simple prompt without context"""
        choices_text = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(question.choices)
        ])
        
        return f"""Câu hỏi: {question.question}

Các lựa chọn:
{choices_text}

Hãy chọn đáp án đúng nhất (A, B, C hoặc D)."""
    
    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract answer letter from LLM response"""
        response_upper = response.upper()
        
        # Look for patterns like "Đáp án: A", "Answer: A", etc.
        for answer_char in ['A', 'B', 'C', 'D']:
            if f"ĐÁP ÁN: {answer_char}" in response_upper:
                return answer_char
            if f"ANSWER: {answer_char}" in response_upper:
                return answer_char
            if f"LỰA CHỌN: {answer_char}" in response_upper:
                return answer_char
        
        # Look for standalone answer at start
        words = response_upper.split()
        if words and words[0] in ['A', 'B', 'C', 'D']:
            return words[0]
        
        # Look for any A, B, C, D in response
        for char in response_upper:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        # Default
        return 'A'
    
    async def process_batch(
        self,
        questions: List[Question],
        batch_size: int = 1,
    ) -> List[PredictionResult]:
        """
        Process batch of questions
        
        Args:
            questions: List of questions
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"Processing {i+1}/{len(questions)} ({question.qid})...", end=" ")
            
            try:
                result = await self.process_question(question)
                results.append(result)
                print(f"✓ Answer: {result.predicted_answer}")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                # Default result on error
                results.append(PredictionResult(
                    qid=question.qid,
                    predicted_answer='A',
                ))
        
        return results
    
    def save_predictions(
        self,
        predictions: List[PredictionResult],
        output_file: str,
    ) -> None:
        """Save predictions to JSON"""
        data = []
        for pred in predictions:
            data.append({
                'qid': pred.qid,
                'predicted_answer': pred.predicted_answer,
            })
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Predictions saved to {output_file}")


async def run_runtime_pipeline(
    questions: List[Question],
    llm_service: LLMService,
    output_file: str,
    config: Optional[Config] = None,
) -> List[PredictionResult]:
    """
    Run complete runtime pipeline
    
    Args:
        questions: List of questions to process
        llm_service: LLM service
        output_file: Output file path
        config: Configuration
        
    Returns:
        List of predictions
    """
    pipeline = RuntimeInferencePipeline(llm_service, config)
    
    print("Starting inference...")
    predictions = await pipeline.process_batch(questions)
    
    print(f"\nProcessed {len(predictions)} predictions")
    pipeline.save_predictions(predictions, output_file)
    
    return predictions

