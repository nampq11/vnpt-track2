"""Main prediction script for VNPT Track 2 QA task"""
import asyncio
import argparse
from pathlib import Path

from src.brain.inference.pipeline import run_pipeline
from src.brain.inference.simple_test import SimpleInferenceTest
from src.brain.inference.processor import QuestionProcessor
from src.brain.llm.services.ollama import OllamaService
from src.brain.llm.services.azure import AzureService


async def main():
    parser = argparse.ArgumentParser(description="VNPT Track 2 - QA Inference Pipeline")
    parser.add_argument(
        "--mode",
        choices=["test", "eval", "inference"],
        default="test",
        help="Running mode: test (single Q test), eval (full dataset eval), inference (predict only)"
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
        help="Ollama model to use"
    )
    parser.add_argument(
        "--llm-service",
        default="ollama",
        choices=["ollama", "azure"],
        help="LLM service to use"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions to test (for test mode)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.mode in ["eval", "inference"]:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM service based on choice
    llm_service = None
    if args.llm_service == "ollama":
        llm_service = OllamaService(
            model=args.model
        )
    elif args.llm_service == "azure":
        llm_service = AzureService(
            model="gpt-4o-mini"
        )
    
    if args.mode == "test":
        # Quick test on first N questions
        print(f"Running quick test on first {args.n} questions...")
        await SimpleInferenceTest.test_first_n_questions(
            file_path=args.input,
            n=args.n,
            model=args.model
        )
    
    elif args.mode == "eval":
        # Full evaluation with metrics
        print("Running full evaluation with metrics...")
        await run_pipeline(
            test_file=args.input,
            output_file=args.output,
            evaluate=True,
            llm_service=llm_service
        )
    
    elif args.mode == "inference":
        # Inference without evaluation
        print("Running inference (no evaluation)...")
        await run_pipeline(
            test_file=args.input,
            output_file=args.output,
            evaluate=False,
            llm_service=llm_service
        )


if __name__ == "__main__":
    asyncio.run(main())
