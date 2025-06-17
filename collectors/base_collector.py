"""
Base collector class for all data sources.
Provides common functionality for rate limiting, retrying, and data validation.
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator
from pathlib import Path

import aiohttp
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry

from schemas import Document, DocumentMetadata, QualityScores


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, calls: int = 100, period: int = 60):
        self.calls = calls
        self.period = period
        self.semaphore = asyncio.Semaphore(calls)
        self.reset_time = asyncio.get_event_loop().time() + period
        
    async def acquire(self):
        async with self.semaphore:
            current_time = asyncio.get_event_loop().time()
            if current_time >= self.reset_time:
                self.reset_time = current_time + self.period
                self.semaphore = asyncio.Semaphore(self.calls)
            await asyncio.sleep(0.1)  # Small delay between calls


class BaseCollector(ABC):
    """Abstract base class for all collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(
            calls=config.get('rate_limit', 100),
            period=config.get('rate_period', 60)
        )
        self.collected_count = 0
        self.error_count = 0
        
        # Setup logging
        logger.add(
            f"logs/{self.__class__.__name__}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=self.config.get('headers', {}),
            timeout=timeout
        )
        logger.info(f"Started {self.__class__.__name__}")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        logger.info(
            f"Stopped {self.__class__.__name__}. "
            f"Collected: {self.collected_count}, Errors: {self.error_count}"
        )
    
    @abstractmethod
    async def extract(self) -> AsyncIterator[Dict[str, Any]]:
        """Extract raw data from source. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def parse(self, raw_data: Any) -> Optional[Document]:
        """Parse raw data into Document format. Must be implemented by subclasses."""
        pass
    
    def validate(self, document: Document) -> bool:
        """Validate document meets quality standards."""
        # Basic validation rules
        if not document.content.text or len(document.content.text) < 100:
            return False
            
        if document.content.word_count < 50:
            return False
            
        if not document.metadata.game.title:
            return False
            
        return True
    
    def generate_document_id(self, content: str) -> str:
        """Generate unique document ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_with_retry(self, url: str, **kwargs) -> Dict[str, Any]:
        """Fetch URL with exponential backoff retry."""
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            self.error_count += 1
            raise
    
    async def process_batch(
        self, 
        items: List[Any], 
        batch_size: int = 10
    ) -> List[Document]:
        """Process items in parallel batches."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            tasks = [self.process_item(item) for item in batch]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Document):
                    results.append(result)
                    self.collected_count += 1
                elif isinstance(result, Exception):
                    logger.error(f"Error processing item: {result}")
                    self.error_count += 1
                    
        return results
    
    async def process_item(self, item: Any) -> Optional[Document]:
        """Process a single item into a Document."""
        try:
            # Extract raw data
            raw_data = await self.extract_item(item)
            
            # Parse into Document
            document = self.parse(raw_data)
            
            if document and self.validate(document):
                return document
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing item {item}: {e}")
            raise
    
    @abstractmethod
    async def extract_item(self, item: Any) -> Any:
        """Extract data for a single item. Must be implemented by subclasses."""
        pass
    
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_file: str):
        """Save progress checkpoint."""
        checkpoint_path = Path(f"checkpoints/{checkpoint_file}")
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'collected_count': self.collected_count,
                'error_count': self.error_count,
                'state': state
            }, f, indent=2)
            
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict[str, Any]]:
        """Load progress checkpoint if exists."""
        checkpoint_path = Path(f"checkpoints/{checkpoint_file}")
        
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint from {checkpoint['timestamp']}")
                return checkpoint.get('state')
                
        return None
    
    def chunk_list(self, lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]