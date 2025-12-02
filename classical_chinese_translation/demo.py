#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translation Demo

Interactive demonstration of the classical Chinese to modern Chinese
translation system using Qwen3-Embedding model.

Features:
- Interactive translation mode
- Batch translation
- Similar text search
- Quick test mode
"""
import os
import sys
import argparse

# Ensure module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from translator import ClassicalChineseTranslator, print_translation_result, TranslationResult


def run_interactive_mode(translator: ClassicalChineseTranslator):
    """
    Run interactive translation mode.
    
    Args:
        translator: Initialized translator instance
    """
    print("\n" + "=" * 70)
    print("🎯 Interactive Classical Chinese Translation")
    print("=" * 70)
    print("\nCommands:")
    print("  - Enter classical Chinese text to translate")
    print("  - Type 'similar <text>' to find similar texts")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("\n📜 Enter classical Chinese text: ").strip()
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n👋 Thank you for using! Goodbye!")
                break
            
            # Check for similar text search
            if user_input.lower().startswith('similar '):
                query = user_input[8:].strip()
                if query:
                    print(f"\n🔍 Finding similar texts to: {query}")
                    similar = translator.get_similar_texts(query, top_k=5)
                    print("-" * 50)
                    for i, item in enumerate(similar, 1):
                        print(f"\n{i}. 《{item['title']}》 (Similarity: {item['similarity']:.2%})")
                        print(f"   Classical: {item['classical']}")
                        print(f"   Modern: {item['modern'][:100]}...")
                continue
            
            # Translate
            print("\n🔄 Translating...")
            result = translator.translate(user_input)
            print_translation_result(result, show_matches=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def run_test_mode(translator: ClassicalChineseTranslator):
    """
    Run quick test mode with predefined examples.
    
    Args:
        translator: Initialized translator instance
    """
    print("\n" + "=" * 70)
    print("🧪 Quick Test Mode")
    print("=" * 70)
    
    test_cases = [
        # Poem lines
        ("梳洗罢，独倚望江楼。", "《望江南》- Famous ci poem"),
        ("过尽千帆皆不是，斜晖脉脉水悠悠。", "《望江南》- Continuation"),
        ("不患寡而患不均，不患贫而患不安。", "《季氏将伐颛臾》- Confucian quote"),
        # Short phrases
        ("学而时习之", "《论语》fragment (if available)"),
        # Longer text
        ("金陵酒肆留别", "Poem title search"),
    ]
    
    for text, description in test_cases:
        print(f"\n📝 Test: {description}")
        print(f"   Input: {text}")
        
        result = translator.translate(text)
        
        print(f"   Translation: {result.translation[:100]}...")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Method: {result.method}")
        print("-" * 50)


def run_batch_mode(translator: ClassicalChineseTranslator, input_file: str, output_file: str):
    """
    Run batch translation mode.
    
    Args:
        translator: Initialized translator instance
        input_file: Path to input file (one text per line)
        output_file: Path to output file
    """
    print(f"\n📂 Batch translating from: {input_file}")
    
    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"   Found {len(texts)} texts to translate")
    
    # Translate
    results = translator.batch_translate(texts)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (text, result) in enumerate(zip(texts, results), 1):
            f.write(f"=== {i} ===\n")
            f.write(f"Original: {text}\n")
            f.write(f"Translation: {result.translation}\n")
            f.write(f"Confidence: {result.confidence:.2%}\n")
            f.write(f"Method: {result.method}\n")
            f.write("\n")
    
    print(f"✅ Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Classical Chinese Translation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Interactive mode
  python demo.py --mode test        # Quick test mode
  python demo.py --mode batch -i input.txt -o output.txt  # Batch mode
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'test', 'batch'],
        default='interactive',
        help='Demo mode (default: interactive)'
    )
    parser.add_argument(
        '-i', '--input',
        help='Input file for batch mode'
    )
    parser.add_argument(
        '-o', '--output',
        default='translations.txt',
        help='Output file for batch mode (default: translations.txt)'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild indexes'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🏛️  Classical Chinese Intelligent Translation System")
    print("    Based on Qwen3-Embedding-4B Model")
    print("=" * 70)
    
    # Initialize translator
    translator = ClassicalChineseTranslator(auto_load=not args.rebuild)
    
    if args.rebuild:
        print("\n🔄 Rebuilding indexes...")
        translator.build_indexes(force_rebuild=True)
    
    # Run selected mode
    if args.mode == 'test':
        run_test_mode(translator)
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Batch mode requires --input file")
            return
        run_batch_mode(translator, args.input, args.output)
    else:
        run_interactive_mode(translator)


if __name__ == "__main__":
    main()
