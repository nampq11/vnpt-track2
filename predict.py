"""Main prediction script for VNPT Track 2 QA task - Titan Shield RAG System"""
import asyncio
import argparse
import json
from pathlib import Path

from src.core.config import Config
from src.core.models import Question
from src.runtime.pipeline import run_runtime_pipeline
from src.runtime.llm.vnpt_service import VNPTService
from src.brain.llm.services.ollama import OllamaService
from src.brain.llm.services.azure import AzureService
from src.brain.inference.evaluator import Evaluator


async def main():
    parser = argparse.ArgumentParser(description="VNPT Track 2 - Titan Shield RAG Inference")
    parser.add_argument(
        "--mode",
        choices=["test", "eval", "inference"],
        default="inference",
        help="Mode: test (5 Q), eval (metrics), inference (predictions only)"
    )
    parser.add_argument(
        "--input",
        default="data/test.json",
        help="Input test file path"
    )
    parser.add_argument(
        "--output",
        default="results/predictions.json",
        help="Output predictions file path"
    )
    parser.add_argument(
        "--model",
        default="qwen3:1.7b",
        help="Model to use (for Ollama)"
    )
    parser.add_argument(
        "--llm-service",
        default="ollama",
        choices=["ollama", "azure", "vnpt"],
        help="LLM service to use"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions for test mode"
    )
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config.from_env()
    
    # Initialize LLM service
    if args.llm_service == "ollama":
        llm_service = OllamaService(
            model=args.model,
        )
    elif args.llm_service == "azure":
        llm_service = AzureService(
            model="gpt-4o-mini",
        )
    elif args.llm_service == "vnpt":
        llm_service = VNPTService(config.vnpt)
    else:
        raise ValueError(f"Unknown LLM service: {args.llm_service}")
    
    # Load questions
    print(f"Loading questions from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit for test mode
    if args.mode == "test":
        data = data[:args.n]
    
    questions = [
        Question(
            qid=item['qid'],
            question=item['question'],
            choices=item['choices'],
            answer=item.get('answer', ''),
        )
        for item in data
    ]
    
    print(f"Loaded {len(questions)} questions")
    
    # Run inference
    print("\nStarting inference pipeline...")
    predictions = await run_runtime_pipeline(
        questions=questions,
        llm_service=llm_service,
        output_file=args.output,
        config=config,
    )
    
    # Evaluate if requested
    if args.mode == "eval":
        print("\nEvaluating predictions...")
        ground_truth = [q.answer for q in questions]
        pred_dicts = [
            {'qid': p.qid, 'predicted_answer': p.predicted_answer}
            for p in predictions
        ]
        
        metrics = Evaluator.evaluate(pred_dicts, ground_truth)
        Evaluator.print_summary(metrics)
        
        # Save metrics
        metrics_file = args.output.replace('.json', '_metrics.json')
        Evaluator.save_results(metrics, metrics_file)
        print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    asyncio.run(main())
