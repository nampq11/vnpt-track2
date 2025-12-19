"""Main inference pipeline for question answering"""
import asyncio
import csv
import time
from typing import List, Optional, Literal
from dataclasses import asdict
import json
from pathlib import Path
from loguru import logger

from src.brain.llm.services.type import LLMService
from src.brain.llm.services.factory import LLMFactory
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
        verbose: bool = False
    ):
        self.llm_service = llm_service
        self.use_agent = use_agent
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.processor = QuestionProcessor()
        self.verbose = verbose
        # Initialize Agent if enabled
        if self.use_agent:
            self.agent = Agent(llm_service=llm_service, verbose=verbose)
    
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
        verbose: bool = False
    ) -> List[PredictionResult]:
        """
        Run inference on a batch of questions concurrently.
        
        Args:
            questions: List of Question objects
            batch_size: Number of parallel requests
        
        Returns:
            List of PredictionResult objects
        """
        semaphore = asyncio.Semaphore(batch_size)
        predictions = []
        total = len(questions)
        completed = 0
        
        print(f"Processing {total} questions with batch size {batch_size}..., verbose: {self.verbose}")
        
        async def process_one(question: Question) -> PredictionResult:
            nonlocal completed
            async with semaphore:
                start_time = time.time()
                try:
                    if self.use_agent:
                        # Format choices as dict
                        options = {
                            chr(65 + i): choice 
                            for i, choice in enumerate(question.choices)
                        }
                        
                        # Call agent
                        result = await self.agent.process_query(
                            query=question.question,
                            options=options,
                            query_id=question.qid,
                            verbose=verbose
                        )
                        answer = result.get('answer', 'A')
                        
                    else:
                        # Use simple LLM prompting (legacy)
                        user_prompt = self.processor.format_for_llm(question)
                        # Pass system prompt separately
                        response = await self.llm_service.generate(
                            user_input=user_prompt,
                            system_message=self.system_prompt
                        )
                        answer = self.processor.parse_answer(response)
                    
                    inference_time = time.time() - start_time
                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        print(f"Progress: {completed}/{total} ({completed/total:.1%})")
                    
                    return PredictionResult(
                        qid=question.qid,
                        predicted_answer=answer,
                        inference_time=inference_time
                    )
                    
                except Exception as e:
                    inference_time = time.time() - start_time
                    logger.error(f"Error processing question {question.qid}: {e}")
                    completed += 1
                    return PredictionResult(
                        qid=question.qid,
                        predicted_answer='A',
                        inference_time=inference_time
                    )

        tasks = [process_one(q) for q in questions]
        predictions = await asyncio.gather(*tasks)
        
        # Sort predictions to match input order (gather preserves order, but just to be safe if logic changes)
        # Actually gather preserves order of tasks list, so this matches 'questions' list order.
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
    
    def save_predictions_time_csv(
        self,
        predictions: List[PredictionResult],
        output_file: str
    ) -> None:
        """Save predictions with time to CSV file with qid,answer,time header"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['qid', 'answer', 'time'])
            for pred in predictions:
                writer.writerow([pred.qid, pred.predicted_answer, f"{pred.inference_time:.4f}"])
        
        print(f"\nPredictions with time saved to {output_file}")

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
    n: Optional[int] = None,
    qids: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    verbose: bool = False
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
        n: Number of questions to process (optional, processes all if not provided)
        qids: List of specific question IDs to process (optional, processes all if not provided)
    
    Returns:
        EvaluationMetrics if evaluate=True, else None
    """
    dataset_name = Path(test_file).stem
    
    # Initialize LLM service via Factory
    llm_service = LLMFactory.create(provider=provider, model=model)
    
    # Create pipeline
    pipeline = InferencePipeline(
        llm_service=llm_service,
        use_agent=use_agent,
        system_prompt=system_prompt,
        verbose=verbose
    )
    
    # Load questions
    print(f"Loading questions from {test_file}...")
    all_questions = pipeline.processor.load_questions(test_file)
    total_count = len(all_questions)
    
    # Filter by specific QIDs if provided
    if qids is not None and len(qids) > 0:
        qid_set = set(qids)
        questions = [q for q in all_questions if q.qid in qid_set]
        if len(questions) == 0:
            print(f"⚠️  Warning: No questions found with specified QIDs: {qids}")
            return None
        print(f"✓ Loaded {len(questions)} questions from '{dataset_name}' (filtered by QIDs: {', '.join(qids)})")
    # Otherwise limit to first n questions if specified
    elif n is not None and n > 0:
        questions = all_questions[:n]
        print(f"✓ Loaded {len(questions)} questions from '{dataset_name}' (limited from {total_count})")
    else:
        questions = all_questions
        print(f"✓ Loaded {len(questions)} questions from '{dataset_name}'")
    print(f"  Output file: {output_file}\n")
    
    # Run inference
    print("Starting inference...")
    # Determine batch size: use provided value, or auto-detect based on provider
    if batch_size is None:
        # Default: 5 for API providers, 1 for local Ollama
        # BTC recommends 4-8 threads for optimal speed
        batch_size = 5 if provider in ["vnpt", "azure"] else 1
    else:
        # Validate batch size (BTC recommends 4-8)
        if batch_size < 1:
            logger.warning(f"Invalid batch_size {batch_size}, using default")
            batch_size = 5 if provider in ["vnpt", "azure"] else 1
        elif batch_size > 8:
            logger.warning(f"Batch size {batch_size} exceeds BTC recommendation (4-8), may cause slower inference")
    
    print(f"Using batch size: {batch_size} threads")
    predictions = await pipeline.run_inference(questions, batch_size=batch_size, verbose=verbose)
    
    # Save predictions
    pipeline.save_predictions(predictions, output_file)
    
    # If output is CSV, also save submission_time.csv (BTC requirement)
    if output_file.endswith('.csv'):
        time_output_file = output_file.replace('.csv', '_time.csv')
        # For submission, use fixed name submission_time.csv
        if output_file == 'submission.csv':
            time_output_file = 'submission_time.csv'
        pipeline.save_predictions_time_csv(predictions, time_output_file)
    
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
