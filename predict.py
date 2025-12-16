"""Main prediction script for VNPT Track 2 QA task"""
import asyncio
import argparse
from pathlib import Path

from src.brain.inference.pipeline import run_pipeline
from src.brain.inference.simple_test import SimpleInferenceTest
from src.brain.inference.processor import QuestionProcessor


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
        default=None,
        help="Model to use (default: vnptai-hackathon-small for VNPT, qwen3:1.7b for Ollama)"
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "vnpt", "azure"],
        default="vnpt",
        help="LLM provider to use (default: vnpt)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions to test (for test mode)"
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        default=False,
        help="Use Agent with task routing (default: False, uses simple prompting)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.mode in ["eval", "inference"]:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "test":
        # Quick test on first N questions
        print(f"Running quick test on first {args.n} questions using {args.provider}...")
        await SimpleInferenceTest.test_first_n_questions(
            file_path=args.input,
            n=args.n,
            model=args.model,
            provider=args.provider
        )
    
    elif args.mode == "eval":
        # Full evaluation with metrics
        mode_str = "Agent" if args.use_agent else "Simple"
        print(f"Running full evaluation ({mode_str} mode) with metrics using {args.provider}...")
        await run_pipeline(
            test_file=args.input,
            output_file=args.output,
            evaluate=True,
            use_agent=args.use_agent,
            provider=args.provider,
            model=args.model
        )
    
    elif args.mode == "inference":
        # Inference without evaluation
        mode_str = "Agent" if args.use_agent else "Simple"
        print(f"Running inference ({mode_str} mode, no evaluation) using {args.provider}...")
        await run_pipeline(
            test_file=args.input,
            output_file=args.output,
            evaluate=False,
            use_agent=args.use_agent,
            provider=args.provider,
            model=args.model
        )


if __name__ == "__main__":
    asyncio.run(main())
