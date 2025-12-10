"""Main inference pipeline for question answering"""
import asyncio
from typing import List, Optional, Dict
from dataclasses import asdict
import json

from src.brain.llm.services.ollama import OllamaService
from src.brain.inference.processor import Question, QuestionProcessor, PredictionResult
from src.brain.inference.evaluator import Evaluator, EvaluationMetrics


class InferencePipeline:
    """Main pipeline for running inference on test data"""
    
    def __init__(
        self,
        llm_service: OllamaService,
        system_prompt: Optional[str] = None,
    ):
        self.llm_service = llm_service
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.processor = QuestionProcessor()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for QA"""
        return """Bạn là một trợ lý thông minh chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.
Hãy đọc kỹ câu hỏi, phân tích các lựa chọn, và chọn đáp án chính xác nhất.
Trước khi đưa ra kết luận, hãy suy luận từng bước.
Cuối cùng, cho biết rõ ràng đáp án bạn chọn (A, B, C hoặc D)."""
    
    async def run_inference(
        self,
        questions: List[Question],
        batch_size: int = 1,
    ) -> List[PredictionResult]:
        """
        Run inference on a batch of questions
        
        Args:
            questions: List of Question objects
            batch_size: Number of parallel requests
        
        Returns:
            List of PredictionResult objects
        """
        predictions = []
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)} ({question.qid})...", end=" ")
            
            try:
                # Format question for LLM
                user_prompt = self.processor.format_for_llm(question)
                
                # Create full message with system prompt
                full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
                
                # Call LLM
                response = await self.llm_service.generate(full_prompt)
                
                # Parse answer
                answer = self.processor.parse_answer(response)
                
                result = PredictionResult(
                    qid=question.qid,
                    predicted_answer=answer
                )
                predictions.append(result)
                print(f"✓ Answer: {answer}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                # Default to 'A' on error
                predictions.append(PredictionResult(
                    qid=question.qid,
                    predicted_answer='A'
                ))
        
        return predictions
    
    def save_predictions(
        self,
        predictions: List[PredictionResult],
        output_file: str
    ) -> None:
        """Save predictions to JSON file"""
        data = [asdict(pred) for pred in predictions]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nPredictions saved to {output_file}")


async def run_pipeline(
    test_file: str,
    output_file: str,
    evaluate: bool = False,
    system_prompt: Optional[str] = None,
) -> Optional[EvaluationMetrics]:
    """
    Run complete inference pipeline
    
    Args:
        test_file: Path to test data JSON file
        output_file: Path to save predictions
        evaluate: Whether to evaluate against ground truth
        system_prompt: Custom system prompt
    
    Returns:
        EvaluationMetrics if evaluate=True, else None
    """
    # Initialize Ollama service
    llm_service = OllamaService(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3:1.7b"
    )
    
    # Create pipeline
    pipeline = InferencePipeline(
        llm_service=llm_service,
        system_prompt=system_prompt
    )
    
    # Load questions
    print(f"Loading questions from {test_file}...")
    questions = pipeline.processor.load_questions(test_file)
    print(f"Loaded {len(questions)} questions\n")
    
    # Run inference
    print("Starting inference...")
    predictions = await pipeline.run_inference(questions)
    
    # Save predictions
    pipeline.save_predictions(predictions, output_file)
    
    # Evaluate if requested
    if evaluate:
        print("\nEvaluating predictions...")
        ground_truth = [q.answer for q in questions]
        pred_dicts = [
            {
                'qid': p.qid,
                'predicted_answer': p.predicted_answer
            }
            for p in predictions
        ]
        
        metrics = Evaluator.evaluate(pred_dicts, ground_truth)
        Evaluator.print_summary(metrics)
        Evaluator.save_results(metrics, output_file.replace('.json', '_metrics.json'))
        
        return metrics
    
    return None

