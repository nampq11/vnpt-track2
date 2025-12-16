#!/usr/bin/env python3
"""Test agent tasks with sample questions from val.json"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.brain.llm.services.vnpt import VNPTService
from src.brain.agent.agent import Agent
from src.brain.agent.query_classification import QueryClassificationService
from loguru import logger


async def load_sample_questions(file_path: str, num_samples: int = 5) -> list:
    """Load first N questions from val.json"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:num_samples]


def format_choices(choices: list) -> Dict[str, str]:
    """Convert choices list to dict with A, B, C, D keys"""
    letters = ['A', 'B', 'C', 'D']
    return {letter: choice for letter, choice in zip(letters, choices)}


async def test_agent_tasks():
    """Test agent with sample questions"""
    print("\n" + "="*70)
    print("AGENT TASK TESTING - Sample from val.json")
    print("="*70)
    
    try:
        # Initialize VNPT service
        print("\n[1/3] Initializing VNPT LLM Service...")
        llm_service = VNPTService(
            model="vnptai-hackathon-small",
            model_type="small"
        )
        print("✓ VNPT Service initialized")
        
        # Initialize agent
        print("[2/3] Initializing Agent...")
        agent = Agent(llm_service=llm_service)
        print("✓ Agent initialized")
        
        # Load sample questions
        print("[3/3] Loading sample questions from val.json...")
        samples = await load_sample_questions("data/val.json", num_samples=5)
        print(f"✓ Loaded {len(samples)} questions")
        
        # Test classification first
        print("\n" + "-"*70)
        print("TESTING QUERY CLASSIFICATION")
        print("-"*70)
        
        classification_service = QueryClassificationService(llm_service=llm_service)
        
        test_cases = [
            ("Điện trở tương đương khi hai điện trở R1 và R2 được mắc song song là gì?", "MATH"),
            ("Đoạn thông tin:\n[1] Tiêu đề: Khỉ thí nghiệm...", "READING"),
            ("Ngôi chùa Ba La Mật được khai dựng vào năm nào?", "RAG"),
        ]
        
        for query, expected_category in test_cases:
            print(f"\nQuery: {query[:60]}...")
            classification = await classification_service.invoke(query)
            print(f"Classification: {classification.get('category', 'UNKNOWN')}")
            print(f"Expected: {expected_category}")
            if classification.get('category') == expected_category:
                print("✓ Correct classification")
            else:
                print("✗ Wrong classification")
        
        # Test agent with specific task types
        print("\n" + "-"*70)
        print("TESTING AGENT TASKS")
        print("-"*70)
        
        for i, sample in enumerate(samples[:3]):  # Test first 3
            qid = sample['qid']
            question = sample['question']
            choices = sample['choices']
            expected_answer = sample['answer']
            
            print(f"\n[Question {i+1}/3] {qid}")
            print(f"Question: {question[:80]}...")
            print(f"Choices: {choices}")
            print(f"Expected: {expected_answer}")
            
            try:
                # Format choices as dict
                options = format_choices(choices)
                
                # Process query through agent
                result = await agent.process_query(
                    query=question,
                    options=options,
                    query_id=qid
                )
                
                predicted_answer = result.get('answer', 'ERROR')
                is_correct = predicted_answer.upper() == expected_answer.upper()
                
                print(f"Predicted: {predicted_answer}")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                logger.error(f"Error processing {qid}: {e}")
        
        print("\n" + "="*70)
        print("TESTING COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


async def test_individual_tasks():
    """Test individual tasks directly"""
    print("\n" + "="*70)
    print("INDIVIDUAL TASK TESTING")
    print("="*70)
    
    try:
        llm_service = VNPTService(
            model="vnptai-hackathon-small",
            model_type="small"
        )
        
        # Test Math Task
        print("\n[MATH TASK]")
        from src.brain.agent.tasks.math import MathTask
        math_task = MathTask(llm_service=llm_service)
        
        math_query = "Điện trở tương đương khi hai điện trở R1 và R2 được mắc song song là gì?"
        math_choices = {"A": "R1 + R2", "B": "R1 - R2", "C": "(R1 * R2) / (R1 + R2)", "D": "(R1 + R2) / (R1 * R2)"}
        
        print(f"Query: {math_query}")
        result = await math_task.invoke(query=math_query, options=math_choices)
        print(f"Result: {result}")
        
        # Test Reading Task
        print("\n[READING TASK]")
        from src.brain.agent.tasks.reading import ReadingTask
        reading_task = ReadingTask(llm_service=llm_service)
        
        # Load a reading sample
        with open("data/val.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reading_sample = next((s for s in data if "Đoạn" in s['question'] and len(s['question']) > 500), data[0])
        reading_choices = format_choices(reading_sample['choices'])
        
        print(f"Query: {reading_sample['question'][:100]}...")
        result = await reading_task.invoke(query=reading_sample['question'], options=reading_choices)
        print(f"Result: {result}")
        
        # Test RAG Task
        print("\n[RAG TASK]")
        from src.brain.agent.tasks.rag import RAGTask
        rag_task = RAGTask(llm_service=llm_service)
        
        rag_query = "Ngôi chùa Ba La Mật được khai dựng vào năm nào?"
        rag_choices = {"A": "1886", "B": "1900", "C": "1920", "D": "1930"}
        
        print(f"Query: {rag_query}")
        result = await rag_task.invoke(
            query=rag_query,
            options=rag_choices,
            temporal_constraint=None,
            key_entities=["Ba La Mật", "chùa"]
        )
        print(f"Result: {result}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR in individual task testing: {str(e)}")
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    print("\n" + "🧪 AGENT TASK TEST SUITE" + "\n")
    
    asyncio.run(test_individual_tasks())
    asyncio.run(test_agent_tasks())

