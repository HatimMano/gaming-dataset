"""
Utility functions for text processing and data manipulation.
"""

import re
import unicodedata
from typing import List, Set, Dict, Any, Optional
import ftfy
from loguru import logger


def clean_text(text: str) -> str:
    """Clean and normalize text.
    
    - Fix encoding issues
    - Normalize unicode
    - Remove excessive whitespace
    - Remove non-printable characters
    """
    # Fix text encoding issues
    text = ftfy.fix_text(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    
    # Remove control characters but keep newlines and tabs
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Strip line starts/ends
    
    return text.strip()


def extract_game_info_from_title(title: str) -> Dict[str, Any]:
    """Extract game information from a title string.
    
    Handles formats like:
    - "Game Title (2023 video game)"
    - "Game Title - Platform Edition"
    - "Game Title: Subtitle"
    """
    # Remove common suffixes
    clean_title = re.sub(r'\s*\([^)]*\)\s*$', '', title)
    clean_title = re.sub(r'\s*-\s*(Edition|Version|Remaster|Remake)\s*$', '', clean_title, flags=re.IGNORECASE)
    
    # Extract year if present
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', title)
    year = int(year_match.group(1)) if year_match else None
    
    # Extract platform if mentioned
    platform_match = re.search(r'\b(PC|PlayStation|PS\d|Xbox|Nintendo|Switch|Mobile|iOS|Android)\b', title, re.IGNORECASE)
    platform = platform_match.group(1) if platform_match else None
    
    return {
        'clean_title': clean_title.strip(),
        'year': year,
        'platform': platform
    }


def load_gaming_dictionary() -> Set[str]:
    """Load gaming-specific terminology dictionary."""
    # This is a subset - in production, load from a comprehensive file
    gaming_terms = {
        # Game mechanics
        'health', 'mana', 'stamina', 'damage', 'dps', 'heal', 'tank', 'support',
        'cooldown', 'buff', 'debuff', 'nerf', 'cc', 'aoe', 'dot', 'hot',
        'crit', 'proc', 'aggro', 'kite', 'gank', 'grief', 'camp',
        
        # Game types
        'fps', 'rpg', 'mmorpg', 'moba', 'rts', 'roguelike', 'roguelite',
        'platformer', 'metroidvania', 'soulslike', 'battle royale',
        
        # Gaming culture
        'noob', 'pro', 'gg', 'glhf', 'elo', 'mmr', 'meta', 'op', 'broken',
        'cheese', 'toxic', 'smurf', 'main', 'alt', 'grind', 'farm',
        'speedrun', 'glitch', 'exploit', 'mod', 'dlc', 'season pass',
        
        # Technical
        'fps', 'ping', 'lag', 'latency', 'hitbox', 'hitscan', 'projectile',
        'raycast', 'texture', 'shader', 'polygon', 'voxel', 'sprite',
        
        # Platforms/Services
        'steam', 'epic', 'origin', 'uplay', 'battlenet', 'xbox live', 'psn',
        'nintendo', 'twitch', 'discord'
    }
    
    # Add variations (plural, past tense, etc.)
    expanded_terms = set()
    for term in gaming_terms:
        expanded_terms.add(term)
        expanded_terms.add(term + 's')  # Simple plural
        expanded_terms.add(term + 'ed')  # Past tense
        expanded_terms.add(term + 'ing')  # Present continuous
        
    return expanded_terms


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using Jaccard similarity."""
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for processing."""
    if len(text) <= max_chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Find end position
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end
            sentence_end = text.rfind('. ', start, end)
            if sentence_end > start:
                end = sentence_end + 1
                
        chunks.append(text[start:end])
        
        # Move start position with overlap
        start = end - overlap
        
    return chunks


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    # Simple sentence splitter - can be enhanced with spaCy
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate text patterns."""
    # Remove navigation elements
    text = re.sub(r'^(Home|Navigate|Menu|Search).*$', '', text, flags=re.MULTILINE)
    
    # Remove "Retrieved from" citations
    text = re.sub(r'Retrieved from.*$', '', text, flags=re.MULTILINE)
    
    # Remove "This article is about" disclaimers
    text = re.sub(r'^This article is about.*?\.', '', text, flags=re.MULTILINE)
    
    # Remove edit markers
    text = re.sub(r'\[edit\]', '', text)
    
    return text.strip()


def format_file_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create ASCII progress bar."""
    progress = current / total if total > 0 else 0
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"