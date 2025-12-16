"""Main inference pipeline for question answering"""
import asyncio
import csv
from typing import List, Optional, Dict, Literal
from dataclasses import asdict
import json

from src.brain.llm.services.ollama import OllamaService
from src.brain.llm.services.vnpt import VNPTService
from src.brain.llm.services.azure import AzureService
from src.brain.llm.services.type import LLMService
from src.brain.agent.agent import Agent
from src.brain.inference.processor import Question, QuestionProcessor, PredictionResult
from src.brain.inference.evaluator import Evaluator, EvaluationMetrics


class InferencePipeline:
    """Main pipeline for running inference on test data"""
    
    def __init__(
        self,
        llm_service: LLMService,
        use_agent: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.llm_service = llm_service
        self.use_agent = use_agent
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.processor = QuestionProcessor()
        
        # Initialize Agent if enabled
        if self.use_agent:
            self.agent = Agent(llm_service=llm_service)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for QA (only used when use_agent=False)"""
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
                if self.use_agent:
                    # Use Agent with task routing
                    # Format choices as dict
                    options = {
                        chr(65 + i): choice 
                        for i, choice in enumerate(question.choices)
                    }
                    
                    # Call agent
                    result = await self.agent.process_query(
                        query=question.question,
                        options=options,
                        query_id=question.qid
                    )
                    
                    # Extract answer from result
                    answer = result.get('answer', 'A')
                    
                else:
                    # Use simple LLM prompting (legacy)
                    user_prompt = self.processor.format_for_llm(question)
                    full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
                    response = await self.llm_service.generate(full_prompt)
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
    
    def save_predictions_csv(
        self,
        predictions: List[PredictionResult],
        output_file: str
    ) -> None:
        """Save predictions to CSV file with qid,answer header"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['qid', 'answer'])
            for pred in predictions:
                writer.writerow([pred.qid, pred.predicted_answer])
        
        print(f"\nPredictions saved to {output_file}")

    def save_predictions(
        self,
        predictions: List[PredictionResult],
        output_file: str
    ) -> None:
        """Save predictions to JSON or CSV file (auto-detect by extension)"""
        if output_file.endswith('.csv'):
            self.save_predictions_csv(predictions, output_file)
        else:
            # Default to JSON format
            data = [asdict(pred) for pred in predictions]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"\nPredictions saved to {output_file}")


async def run_pipeline(
    test_file: str,
    output_file: str,
    evaluate: bool = False,
    use_agent: bool = True,
    system_prompt: Optional[str] = None,
    provider: Literal["ollama", "vnpt", "azure"] = "ollama",
    model: Optional[str] = None,
) -> Optional[EvaluationMetrics]:
    """
    Run complete inference pipeline
    
    Args:
        test_file: Path to test data JSON file
        output_file: Path to save predictions
        evaluate: Whether to evaluate against ground truth
        use_agent: Whether to use Agent with task routing (default: True)
        system_prompt: Custom system prompt (only used when use_agent=False)
        provider: LLM provider to use ("ollama", "vnpt", or "azure")
        model: Model name (optional, uses default if not provided)
    
    Returns:
        EvaluationMetrics if evaluate=True, else None
    """
    # Extract dataset name from file path
    from pathlib import Path
    dataset_name = Path(test_file).stem
    
    # Initialize LLM service based on provider
    if provider == "vnpt":
        model_name = model or "vnptai-hackathon-small"
        model_type = "small" if "small" in model_name else "large"
        llm_service = VNPTService(
            model=model_name,
            model_type=model_type
        )
    elif provider == "azure":
        llm_service = AzureService(
            model=model or "gpt-4.1"
        )
    else:  # ollama
        llm_service = OllamaService(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model or "qwen3:1.7b"
        )
    
    # Create pipeline
    pipeline = InferencePipeline(
        llm_service=llm_service,
        use_agent=use_agent,
        system_prompt=system_prompt
    )
    
    # Load questions
    print(f"Loading questions from {test_file}...")
    questions = pipeline.processor.load_questions(test_file)
    print(f"✓ Loaded {len(questions)} questions from '{dataset_name}'")
    print(f"  Output file: {output_file}\n")
    
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
        Evaluator.print_summary(metrics, dataset_name=dataset_name)
        
        # Save detailed metrics
        metrics_file = output_file.replace('.json', '_metrics.json').replace('.csv', '_metrics.json')
        Evaluator.save_results(metrics, metrics_file)
        print(f"✓ Metrics saved to {metrics_file}")
        
        return metrics
    
    return None

