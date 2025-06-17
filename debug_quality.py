#!/usr/bin/env python3
"""
Debug script to analyze why quality scores are low.
"""

import asyncio
from collectors.wikipedia_collector import WikipediaCollector
from processors.quality_scorer import QualityScorer
from utils import load_gaming_dictionary
import json
from loguru import logger

async def debug_quality():
    """Debug quality scoring issues."""
    # Initialize components
    collector = WikipediaCollector()
    scorer = QualityScorer()
    gaming_terms = load_gaming_dictionary()
    
    # Collect just a few documents for analysis
    docs_analyzed = 0
    
    async with collector:
        async for doc_dict in collector.extract():
            if docs_analyzed >= 3:  # Analyze first 3 documents
                break
                
            # Recreate Document object
            from schemas import Document
            doc = Document.from_dict(doc_dict)
            
            print("\n" + "="*80)
            print(f"DOCUMENT: {doc.content.title}")
            print("="*80)
            
            # Check content
            print(f"\nOriginal text length: {len(doc.content.text)}")
            print(f"Cleaned text length: {len(doc.content.text_clean)}")
            print(f"Word count: {doc.content.word_count}")
            
            # Show first 500 chars of cleaned text
            print(f"\nFirst 500 chars of cleaned text:")
            print(doc.content.text_clean[:500])
            
            # Calculate detailed scores
            text = doc.content.text_clean
            
            # Individual scoring components
            gaming_density = scorer._score_gaming_density(text)
            structure = scorer._score_structure(text)
            uniqueness = scorer._score_uniqueness(text, doc_dict)
            length = scorer._score_length(doc.content.word_count)
            freshness = scorer._score_freshness(doc_dict)
            
            print(f"\nDETAILED SCORES:")
            print(f"Gaming Density (35%): {gaming_density:.3f}")
            print(f"Structure (20%): {structure:.3f}")
            print(f"Uniqueness (20%): {uniqueness:.3f}")
            print(f"Length (15%): {length:.3f}")
            print(f"Freshness (10%): {freshness:.3f}")
            
            # Calculate overall
            overall = doc.calculate_quality_score(gaming_terms)
            print(f"\nOVERALL SCORE: {overall:.3f}")
            
            # Check gaming terms found
            text_lower = text.lower()
            words = text_lower.split()
            gaming_words_found = [word for word in words if word in gaming_terms]
            print(f"\nGaming terms found ({len(gaming_words_found)}): {gaming_words_found[:20]}")
            
            # Check metadata
            print(f"\nGame info: {doc.metadata.game.title}")
            print(f"Genres: {doc.metadata.game.genres}")
            print(f"Platforms: {doc.metadata.classification.platforms}")
            print(f"Content type: {doc.metadata.classification.content_type}")
            
            docs_analyzed += 1

if __name__ == "__main__":
    asyncio.run(debug_quality())