#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Chinese Translation Data Processor

This module handles loading, parsing, and preprocessing of classical Chinese texts
and their modern translations. It supports multi-granularity text segmentation
for improved translation accuracy.
"""
import os
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

from project_config import DATA_DIR, CLASSICAL_SENTENCE_DELIMITERS, MODERN_SENTENCE_DELIMITERS


@dataclass
class TextPair:
    """
    Full text pair containing classical Chinese and modern translation.
    
    Attributes:
        title: Title of the poem/prose
        author: Author name
        classical: Original classical Chinese text
        modern: Modern Chinese translation
        source_file: Path to the source file
    """
    title: str
    author: str
    classical: str
    modern: str
    source_file: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TextPair':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class SentencePair:
    """
    Sentence-level pair for fine-grained translation matching.
    
    Attributes:
        title: Source poem/prose title
        classical: Classical Chinese sentence
        modern: Modern Chinese translation
        sentence_idx: Position index in original text
        full_classical: Complete classical text for context
        full_modern: Complete modern translation for context
    """
    title: str
    classical: str
    modern: str
    sentence_idx: int
    full_classical: str = ""
    full_modern: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SentencePair':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class ClausePair:
    """
    Clause-level pair for ultra-fine-grained matching.
    
    Attributes:
        title: Source poem/prose title
        classical: Classical Chinese clause
        modern: Modern Chinese translation
        clause_idx: Position index
        parent_classical: Parent sentence for context
    """
    title: str
    classical: str
    modern: str
    clause_idx: int
    parent_classical: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class DataProcessor:
    """
    Multi-granularity data processor for classical Chinese translation.
    
    This processor supports three levels of text granularity:
    - Full text: Complete poems/prose
    - Sentence: Individual sentences
    - Clause: Sub-sentence clauses (separated by commas, etc.)
    """
    
    # Clause-level delimiters (finer than sentence)
    CLAUSE_DELIMITERS = ['，', '、', '；', '：']
    
    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the poem data
        """
        self.data_dir = data_dir
        self.text_pairs: List[TextPair] = []
        self.sentence_pairs: List[SentencePair] = []
        self.clause_pairs: List[ClausePair] = []
        
    def load_all_data(self) -> List[TextPair]:
        """
        Load all poem/prose data from the data directory.
        
        Returns:
            List of TextPair objects
        """
        print(f"📚 Loading data from {self.data_dir}...")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.text_pairs = []
        
        for poem_dir in sorted(os.listdir(self.data_dir)):
            poem_path = os.path.join(self.data_dir, poem_dir)
            if not os.path.isdir(poem_path):
                continue
            
            # Try to load from interleaved file first (most accurate)
            interleaved_file = os.path.join(poem_path, "原文译文穿插.txt")
            original_file = os.path.join(poem_path, "原文.txt")
            translation_file = os.path.join(poem_path, "译文.txt")
            
            try:
                if os.path.exists(interleaved_file):
                    # Parse interleaved format for better alignment
                    text_pair = self._parse_interleaved_file(interleaved_file, poem_path)
                elif os.path.exists(original_file) and os.path.exists(translation_file):
                    # Fallback to separate files
                    text_pair = self._parse_separate_files(original_file, translation_file, poem_path)
                else:
                    continue
                
                if text_pair and text_pair.modern != "暂无翻译" and len(text_pair.modern.strip()) >= 10:
                    self.text_pairs.append(text_pair)
                    
            except Exception as e:
                print(f"⚠️ Failed to parse {poem_dir}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(self.text_pairs)} texts")
        return self.text_pairs
    
    def _parse_interleaved_file(self, filepath: str, poem_path: str) -> Optional[TextPair]:
        """
        Parse interleaved format file (original and translation alternating).
        
        This format provides better sentence alignment.
        
        Args:
            filepath: Path to the interleaved file
            poem_path: Directory path for source reference
            
        Returns:
            TextPair object or None if parsing fails
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        title = ""
        author = ""
        classical_parts = []
        modern_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("标题："):
                title = line[3:].strip()
            elif line.startswith("作者："):
                author = line[3:].strip()
            elif line.startswith("译文："):
                # This is a translation line - clean it
                trans_text = line[3:].strip()
                trans_text = self._clean_translation_text(trans_text)
                if trans_text:
                    modern_parts.append(trans_text)
            elif not any(line.startswith(p) for p in ["译文", "注释", "展开"]):
                # This is an original line
                classical_parts.append(line)
        
        if not classical_parts:
            return None
        
        modern_text = '\n'.join(modern_parts) if modern_parts else "暂无翻译"
        modern_text = self._clean_translation_text(modern_text)
            
        return TextPair(
            title=title,
            author=author,
            classical='\n'.join(classical_parts),
            modern=modern_text,
            source_file=poem_path
        )
    
    def _parse_separate_files(self, original_file: str, translation_file: str, 
                              poem_path: str) -> Optional[TextPair]:
        """
        Parse separate original and translation files.
        
        Args:
            original_file: Path to the original text file
            translation_file: Path to the translation file
            poem_path: Directory path for source reference
            
        Returns:
            TextPair object or None if parsing fails
        """
        # Read original
        with open(original_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Read translation
        with open(translation_file, 'r', encoding='utf-8') as f:
            translation_content = f.read()
        
        title, author, classical = self._parse_text_content(original_content)
        _, _, modern = self._parse_translation_content(translation_content)
        
        if not classical:
            return None
            
        return TextPair(
            title=title,
            author=author,
            classical=classical,
            modern=modern,
            source_file=poem_path
        )
    
    def _parse_text_content(self, content: str) -> Tuple[str, str, str]:
        """
        Parse text content to extract title, author, and body.
        
        Args:
            content: Raw text content
            
        Returns:
            Tuple of (title, author, body_text)
        """
        lines = content.strip().split('\n')
        title = ""
        author = ""
        text_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("标题："):
                title = line[3:].strip()
            elif line.startswith("作者："):
                author = line[3:].strip()
            elif line and not any(line.startswith(p) for p in ["译文", "注释", "展开"]):
                text_lines.append(line)
        
        return title, author, '\n'.join(text_lines)
    
    def _parse_translation_content(self, content: str) -> Tuple[str, str, str]:
        """
        Parse translation content, handling various formats.
        
        Args:
            content: Raw translation content
            
        Returns:
            Tuple of (title, author, translation_text)
        """
        lines = content.strip().split('\n')
        title = ""
        author = ""
        text_lines = []
        in_translation = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("标题："):
                title = line[3:].strip()
            elif line.startswith("作者："):
                author = line[3:].strip()
            elif line == "译文及注释" or line == "译文":
                in_translation = True
                continue
            elif line == "注释" or line.startswith("注释"):
                break  # Stop at annotations section
            elif line.startswith("展开阅读全文"):
                break
            elif in_translation and line:
                text_lines.append(line)
            elif not in_translation and line and not any(
                line.startswith(p) for p in ["译文", "注释", "展开"]
            ):
                text_lines.append(line)
        
        # Clean the translation text
        result = '\n'.join(text_lines)
        result = self._clean_translation_text(result)
        
        return title, author, result
    
    def _clean_translation_text(self, text: str) -> str:
        """
        Clean translation text by removing common prefixes and noise.
        
        Args:
            text: Raw translation text
            
        Returns:
            Cleaned translation text
        """
        if not text:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = [
            "译文及注释 译文 ",
            "译文及注释 ",
            "译文 ",
            "翻译 ",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):]
        
        # Remove noise patterns
        noise_patterns = [
            "展开阅读全文 ∨",
            "展开阅读全文",
            "∨",
        ]
        
        for noise in noise_patterns:
            text = text.replace(noise, "")
        
        # Remove trailing annotations marker
        if "注释" in text:
            # Find clean break point
            parts = text.split("注释")
            if parts[0].strip():
                text = parts[0].strip()
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def split_into_sentences(self, text: str, is_classical: bool = True) -> List[str]:
        """
        Split text into sentences using appropriate delimiters.
        
        Args:
            text: Text to split
            is_classical: Whether the text is classical Chinese
            
        Returns:
            List of sentences
        """
        delimiters = CLASSICAL_SENTENCE_DELIMITERS if is_classical else MODERN_SENTENCE_DELIMITERS
        
        # Build regex pattern
        pattern = '([' + ''.join(re.escape(d) for d in delimiters) + '])'
        parts = re.split(pattern, text)
        
        sentences = []
        current = ""
        
        for part in parts:
            if part in delimiters:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if current.strip():
            sentences.append(current.strip())
        
        # Filter out very short segments
        return [s for s in sentences if len(s) >= 2]
    
    def split_into_clauses(self, text: str) -> List[str]:
        """
        Split text into clauses (finer than sentences).
        
        This is useful for matching partial phrases.
        
        Args:
            text: Text to split
            
        Returns:
            List of clauses
        """
        all_delimiters = self.CLAUSE_DELIMITERS + CLASSICAL_SENTENCE_DELIMITERS
        pattern = '([' + ''.join(re.escape(d) for d in all_delimiters) + '])'
        parts = re.split(pattern, text)
        
        clauses = []
        current = ""
        
        for part in parts:
            if part in all_delimiters:
                current += part
                if len(current.strip()) >= 2:  # Minimum clause length
                    clauses.append(current.strip())
                current = ""
            else:
                current += part
        
        if len(current.strip()) >= 2:
            clauses.append(current.strip())
        
        return clauses
    
    def create_sentence_pairs(self) -> List[SentencePair]:
        """
        Create sentence-level pairs with improved alignment.
        
        Uses multiple strategies:
        1. Direct alignment when counts match
        2. Similarity-based alignment when counts differ
        3. Full text fallback for unmatched sentences
        
        Returns:
            List of SentencePair objects
        """
        print("📝 Creating sentence-level pairs...")
        
        if not self.text_pairs:
            self.load_all_data()
        
        self.sentence_pairs = []
        
        for text_pair in self.text_pairs:
            # Split both texts
            classical_sentences = self.split_into_sentences(text_pair.classical, is_classical=True)
            modern_sentences = self.split_into_sentences(text_pair.modern, is_classical=False)
            
            # Try to align sentences
            aligned_pairs = self._align_sentences(
                classical_sentences, 
                modern_sentences,
                text_pair.title,
                text_pair.classical,
                text_pair.modern
            )
            
            self.sentence_pairs.extend(aligned_pairs)
        
        print(f"✅ Created {len(self.sentence_pairs)} sentence pairs")
        return self.sentence_pairs
    
    def _align_sentences(self, classical: List[str], modern: List[str],
                        title: str, full_classical: str, full_modern: str) -> List[SentencePair]:
        """
        Align classical and modern sentences using multiple strategies.
        
        Args:
            classical: List of classical sentences
            modern: List of modern sentences
            title: Source title
            full_classical: Complete classical text
            full_modern: Complete modern translation
            
        Returns:
            List of aligned SentencePair objects
        """
        pairs = []
        
        # Strategy 1: Direct 1-to-1 alignment for matching counts
        if len(classical) == len(modern):
            for i, (c, m) in enumerate(zip(classical, modern)):
                pairs.append(SentencePair(
                    title=title,
                    classical=c,
                    modern=m,
                    sentence_idx=i,
                    full_classical=full_classical,
                    full_modern=full_modern
                ))
            return pairs
        
        # Strategy 2: Proportional alignment
        min_len = min(len(classical), len(modern))
        
        if len(classical) <= len(modern):
            # More modern sentences - combine modern for each classical
            ratio = len(modern) / len(classical) if classical else 1
            for i, c in enumerate(classical):
                start_idx = int(i * ratio)
                end_idx = int((i + 1) * ratio)
                combined_modern = ' '.join(modern[start_idx:end_idx])
                pairs.append(SentencePair(
                    title=title,
                    classical=c,
                    modern=combined_modern,
                    sentence_idx=i,
                    full_classical=full_classical,
                    full_modern=full_modern
                ))
        else:
            # More classical sentences - use full modern for extras
            for i in range(min_len):
                pairs.append(SentencePair(
                    title=title,
                    classical=classical[i],
                    modern=modern[i],
                    sentence_idx=i,
                    full_classical=full_classical,
                    full_modern=full_modern
                ))
            
            # For remaining classical sentences, use nearby modern or full modern
            for i in range(min_len, len(classical)):
                # Use the last modern sentence or full translation
                fallback_modern = modern[-1] if modern else full_modern
                pairs.append(SentencePair(
                    title=title,
                    classical=classical[i],
                    modern=fallback_modern,
                    sentence_idx=i,
                    full_classical=full_classical,
                    full_modern=full_modern
                ))
        
        return pairs
    
    def create_clause_pairs(self) -> List[ClausePair]:
        """
        Create clause-level pairs for fine-grained matching.
        
        Returns:
            List of ClausePair objects
        """
        print("📝 Creating clause-level pairs...")
        
        if not self.sentence_pairs:
            self.create_sentence_pairs()
        
        self.clause_pairs = []
        
        for sp in self.sentence_pairs:
            classical_clauses = self.split_into_clauses(sp.classical)
            modern_clauses = self.split_into_clauses(sp.modern)
            
            # Align clauses within sentence
            min_len = min(len(classical_clauses), len(modern_clauses))
            
            for i in range(min_len):
                self.clause_pairs.append(ClausePair(
                    title=sp.title,
                    classical=classical_clauses[i],
                    modern=modern_clauses[i],
                    clause_idx=i,
                    parent_classical=sp.classical
                ))
        
        print(f"✅ Created {len(self.clause_pairs)} clause pairs")
        return self.clause_pairs
    
    def get_all_classical_texts(self, granularity: str = "sentence") -> List[str]:
        """
        Get all classical texts at specified granularity.
        
        Args:
            granularity: "full", "sentence", or "clause"
            
        Returns:
            List of texts
        """
        if granularity == "full":
            return [tp.classical for tp in self.text_pairs]
        elif granularity == "sentence":
            return [sp.classical for sp in self.sentence_pairs]
        elif granularity == "clause":
            return [cp.classical for cp in self.clause_pairs]
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
    
    def save_processed_data(self, output_file: str):
        """
        Save all processed data to JSON file.
        
        Args:
            output_file: Output file path
        """
        data = {
            'text_pairs': [tp.to_dict() for tp in self.text_pairs],
            'sentence_pairs': [sp.to_dict() for sp in self.sentence_pairs],
            'clause_pairs': [cp.to_dict() for cp in self.clause_pairs]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Data saved to {output_file}")
    
    def load_processed_data(self, input_file: str):
        """
        Load processed data from JSON file.
        
        Args:
            input_file: Input file path
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.text_pairs = [TextPair.from_dict(tp) for tp in data.get('text_pairs', [])]
        self.sentence_pairs = [SentencePair.from_dict(sp) for sp in data.get('sentence_pairs', [])]
        
        if 'clause_pairs' in data:
            self.clause_pairs = [ClausePair(**cp) for cp in data['clause_pairs']]
        
        print(f"✅ Loaded {len(self.text_pairs)} texts, {len(self.sentence_pairs)} sentences")
    
    def print_statistics(self):
        """Print comprehensive data statistics."""
        print("\n" + "=" * 60)
        print("Data Statistics")
        print("=" * 60)
        print(f"Total texts: {len(self.text_pairs)}")
        print(f"Total sentence pairs: {len(self.sentence_pairs)}")
        print(f"Total clause pairs: {len(self.clause_pairs)}")
        
        if self.text_pairs:
            avg_classical_len = sum(len(tp.classical) for tp in self.text_pairs) / len(self.text_pairs)
            avg_modern_len = sum(len(tp.modern) for tp in self.text_pairs) / len(self.text_pairs)
            print(f"Avg classical text length: {avg_classical_len:.1f} chars")
            print(f"Avg modern text length: {avg_modern_len:.1f} chars")
        
        if self.sentence_pairs:
            avg_sentence_len = sum(len(sp.classical) for sp in self.sentence_pairs) / len(self.sentence_pairs)
            print(f"Avg classical sentence length: {avg_sentence_len:.1f} chars")
        
        print("=" * 60)
    
    def get_sample_pairs(self, n: int = 5) -> List[SentencePair]:
        """
        Get sample sentence pairs for inspection.
        
        Args:
            n: Number of samples
            
        Returns:
            List of sample SentencePair objects
        """
        return self.sentence_pairs[:n]


def main():
    """Test the data processor."""
    processor = DataProcessor()
    
    # Load and process data
    processor.load_all_data()
    processor.create_sentence_pairs()
    processor.create_clause_pairs()
    processor.print_statistics()
    
    # Show samples
    print("\n📖 Sample data:")
    print("-" * 60)
    
    if processor.text_pairs:
        sample = processor.text_pairs[0]
        print(f"Title: {sample.title}")
        print(f"Author: {sample.author}")
        print(f"Classical: {sample.classical[:100]}...")
        print(f"Modern: {sample.modern[:100]}...")
    
    print("-" * 60)
    
    if processor.sentence_pairs:
        print("\n📝 Sentence pair samples:")
        for i, sp in enumerate(processor.sentence_pairs[:5]):
            print(f"\n{i+1}. Classical: {sp.classical}")
            print(f"   Modern: {sp.modern[:80]}..." if len(sp.modern) > 80 else f"   Modern: {sp.modern}")


if __name__ == "__main__":
    main()
