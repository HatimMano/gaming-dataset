"""
Base collector class for all data sources.
Provides common functionality for rate limiting, retrying, and data validation.
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator, Union
from pathlib import Path
from dataclasses import dataclass, field

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class CollectorConfig:
    """Base configuration for collectors."""
    rate_limit: int = 10  # calls per period
    rate_period: int = 60  # seconds
    min_content_length: int = 100
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100  # documents
    

class ValidationError(Exception):
    """Raised when document validation fails."""
    pass


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.semaphore = asyncio.Semaphore(calls)
        self.reset_time = None
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make a call."""
        async with self.lock:
            current_time = asyncio.get_event_loop().time()
            
            if self.reset_time is None:
                self.reset_time = current_time + self.period
            elif current_time >= self.reset_time:
                # Reset the semaphore
                self.reset_time = current_time + self.period
                self.semaphore = asyncio.Semaphore(self.calls)
        
        async with self.semaphore:
            # Small delay between calls to be respectful
            await asyncio.sleep(0.1)


class BaseCollector(ABC):
    """Abstract base class for all collectors."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            calls=config.rate_limit,
            period=config.rate_period
        )
        self.stats = {
            'collected_count': 0,
            'error_count': 0,
            'skipped_count': 0,
            'start_time': datetime.utcnow()
        }
        
        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / f"{self.__class__.__name__}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    async def __aenter__(self):
        """Async context manager entry."""
        logger.info(f"Started {self.__class__.__name__}")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        duration = datetime.utcnow() - self.stats['start_time']
        logger.info(
            f"Stopped {self.__class__.__name__}. "
            f"Duration: {duration}, "
            f"Collected: {self.stats['collected_count']}, "
            f"Errors: {self.stats['error_count']}, "
            f"Skipped: {self.stats['skipped_count']}"
        )
    
    @abstractmethod
    async def extract(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Extract documents from source.
        
        Yields:
            Dict containing document data
        """
        pass
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """
        Validate document meets quality standards.
        
        Args:
            document: Document dict to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If document has critical issues
        """
        # Check required fields
        required_fields = ['document_id', 'source', 'content']
        for field in required_fields:
            if field not in document:
                raise ValidationError(f"Missing required field: {field}")
        
        # Check content
        content = document.get('content', {})
        text = content.get('text_clean') or content.get('text', '')
        
        if not text:
            raise ValidationError("No text content found")
            
        if len(text) < self.config.min_content_length:
            return False
            
        return True
    
    def generate_document_id(self, source: str, unique_id: Union[str, int]) -> str:
        """
        Generate unique document ID.
        
        Args:
            source: Source platform (wikipedia, steam, etc.)
            unique_id: Source-specific unique identifier
            
        Returns:
            Unique document ID
        """
        timestamp = int(datetime.utcnow().timestamp())
        return f"{source}_{unique_id}_{timestamp}"
    
    async def save_checkpoint(self, state: Dict[str, Any]):
        """Save progress checkpoint."""
        if not self.config.checkpoint_enabled:
            return
            
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{self.__class__.__name__}_checkpoint.json"
        
        checkpoint_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'stats': self.stats,
            'state': state
        }
        
        # Save atomically
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        temp_file.replace(checkpoint_file)
        logger.debug(f"Saved checkpoint: {self.stats['collected_count']} documents")
    
    async def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load progress checkpoint if exists."""
        if not self.config.checkpoint_enabled:
            return None
            
        checkpoint_file = Path("checkpoints") / f"{self.__class__.__name__}_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                # Restore stats
                self.stats.update(checkpoint.get('stats', {}))
                self.stats['start_time'] = datetime.fromisoformat(
                    self.stats['start_time']
                ) if isinstance(self.stats['start_time'], str) else self.stats['start_time']
                
                logger.info(
                    f"Loaded checkpoint from {checkpoint['timestamp']}, "
                    f"resuming from {self.stats['collected_count']} documents"
                )
                
                return checkpoint.get('state')
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None
                
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_with_retry(
        self, 
        session: aiohttp.ClientSession,
        url: str, 
        **kwargs
    ) -> Union[Dict, str]:
        """
        Fetch URL with exponential backoff retry.
        
        Args:
            session: aiohttp session
            url: URL to fetch
            **kwargs: Additional arguments for session.get()
            
        Returns:
            Response data (JSON dict or text)
        """
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(url, **kwargs) as response:
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    return await response.text()
                    
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            self.stats['error_count'] += 1
            raise
    
    def normalize_text(self, text: str) -> str:
        """
        Basic text normalization.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove multiple newlines
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        
        return text.strip()
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~0.75 words per token for English
        word_count = len(text.split())
        return int(word_count * 0.75)