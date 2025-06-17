"""
Data schemas for the gaming dataset.
Uses dataclasses for type safety and easy serialization.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class ContentType(Enum):
    """Types of gaming content."""
    ENCYCLOPEDIA = "encyclopedia"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    DISCUSSION = "discussion"
    NEWS = "news"
    GUIDE = "guide"
    WIKI = "wiki"


class Platform(Enum):
    """Gaming platforms."""
    PC = "PC"
    PS5 = "PlayStation 5"
    PS4 = "PlayStation 4"
    XBOX_SERIES = "Xbox Series X/S"
    XBOX_ONE = "Xbox One"
    SWITCH = "Nintendo Switch"
    MOBILE = "Mobile"
    VR = "VR"


@dataclass
class GameInfo:
    """Information about a specific game."""
    title: str
    aliases: List[str] = field(default_factory=list)
    game_id: Optional[str] = None
    release_date: Optional[str] = None
    developer: Optional[str] = None
    publisher: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Author:
    """Information about content author."""
    name: Optional[str] = None
    id: Optional[str] = None
    verified: bool = False
    platform: Optional[str] = None


@dataclass
class Source:
    """Source information."""
    platform: str  # wikipedia, steam, youtube, etc.
    url: str
    api_endpoint: Optional[str] = None
    crawl_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Content:
    """Main content of the document."""
    title: str
    text: str
    text_clean: Optional[str] = None
    language: str = "en"
    word_count: int = 0
    char_count: int = 0
    
    def __post_init__(self):
        """Calculate counts after initialization."""
        if self.text:
            self.word_count = len(self.text.split())
            self.char_count = len(self.text)


@dataclass
class QualityScores:
    """Quality assessment scores."""
    overall: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    uniqueness: float = 0.0
    gaming_density: float = 0.0
    structure: float = 0.0
    freshness: float = 0.0
    
    def calculate_overall(self, weights: Optional[Dict[str, float]] = None):
        """Calculate weighted overall score."""
        if weights is None:
            weights = {
                'gaming_density': 0.35,
                'structure': 0.20,
                'uniqueness': 0.20,
                'relevance': 0.15,
                'freshness': 0.10
            }
        
        self.overall = sum(
            getattr(self, key) * weight 
            for key, weight in weights.items()
            if hasattr(self, key)
        )
        return self.overall


@dataclass
class Classification:
    """Content classification."""
    content_type: ContentType
    genres: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_tutorial: bool = False
    is_review: bool = False
    is_news: bool = False


@dataclass
class Processing:
    """Processing metadata."""
    pipeline_version: str = "1.0.0"
    extracted_entities: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    toxicity_score: float = 0.0
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DocumentMetadata:
    """Complete metadata for a document."""
    game: GameInfo
    classification: Classification
    author: Optional[Author] = None


@dataclass
class Document:
    """Main document structure."""
    document_id: str
    source: Source
    content: Content
    metadata: DocumentMetadata
    quality: Optional[QualityScores] = None
    processing: Optional[Processing] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary."""
        # Convert nested dictionaries to dataclasses
        data['source'] = Source(**data['source'])
        data['content'] = Content(**data['content'])
        
        # Handle nested metadata
        game_data = data['metadata']['game']
        data['metadata']['game'] = GameInfo(**game_data)
        
        classification_data = data['metadata']['classification']
        # Convert string to ContentType enum
        classification_data['content_type'] = ContentType(classification_data['content_type'])
        data['metadata']['classification'] = Classification(**classification_data)
        
        if 'author' in data['metadata'] and data['metadata']['author']:
            data['metadata']['author'] = Author(**data['metadata']['author'])
            
        data['metadata'] = DocumentMetadata(**data['metadata'])
        
        # Handle optional fields
        if 'quality' in data and data['quality']:
            data['quality'] = QualityScores(**data['quality'])
            
        if 'processing' in data and data['processing']:
            data['processing'] = Processing(**data['processing'])
            
        return cls(**data)
    
    def calculate_quality_score(self, gaming_terms: set) -> float:
        """Calculate quality score for this document."""
        if not self.quality:
            self.quality = QualityScores()
        
        # Gaming density
        text_lower = self.content.text.lower()
        words = text_lower.split()
        gaming_words = sum(1 for word in words if word in gaming_terms)
        self.quality.gaming_density = min(gaming_words / len(words) * 10, 1.0) if words else 0.0
        
        # Length score
        optimal_length = 500
        length_ratio = self.content.word_count / optimal_length
        self.quality.relevance = min(length_ratio, 1.0) if length_ratio < 2 else 2 / length_ratio
        
        # Structure score (basic - check for paragraphs, lists, etc.)
        self.quality.structure = min(
            len(self.content.text.split('\n\n')) / 5, 1.0
        )
        
        # Calculate overall
        self.quality.calculate_overall()
        
        return self.quality.overall