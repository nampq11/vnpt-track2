"""Simple inference example"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.inference.pipeline import run_pipeline
from src.brain.inference.simple_test import SimpleInferenceTest


async def example_quick_test():
    """Example 1: Quick test on 5 questions"""
    print("="*60)
    print("Example 1: Quick Test (5 questions)")
    print("="*60)
    
    await SimpleInferenceTest.test_first_n_questions(
        file_path="data/test.json",
        n=5,
        model="qwen3:1.7b"
    )


async def example_full_eval():
    """Example 2: Full evaluation"""
    print("\n" + "="*60)
    print("Example 2: Full Evaluation")
    print("="*60)
    
    metrics = await run_pipeline(
        test_file="data/test.json",
        output_file="results/example_predictions.json",
        evaluate=True
    )
    
    if metrics:
        print(f"\nFinal Accuracy: {metrics.accuracy:.2%}")


async def example_inference_only():
    """Example 3: Inference without evaluation"""
    print("\n" + "="*60)
    print("Example 3: Inference Only (no evaluation)")
    print("="*60)
    
    await run_pipeline(
        test_file="data/test.json",
        output_file="results/example_predictions_no_eval.json",
        evaluate=False
    )


async def main():
    """Run examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference examples")
    parser.add_argument(
        "--example",
        choices=["1", "2", "3", "all"],
        default="1",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example in ["1", "all"]:
        try:
            await example_quick_test()
        except Exception as e:
            print(f"Example 1 failed: {e}")
    
    if args.example in ["2", "all"]:
        try:
            await example_full_eval()
        except Exception as e:
            print(f"Example 2 failed: {e}")
    
    if args.example in ["3", "all"]:
        try:
            await example_inference_only()
        except Exception as e:
            print(f"Example 3 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

