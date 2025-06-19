#!/usr/bin/env python3
"""
Wikipedia Gaming Collector using official Wikipedia API.
Optimized for high-quality gaming content extraction.
"""

import asyncio
import wikipediaapi
import wptools
import re
from typing import Dict, List, Optional, AsyncIterator, Tuple
from datetime import datetime
from pathlib import Path
import json
from loguru import logger
import aiohttp
import time
from dataclasses import dataclass, asdict

from .base_collector import BaseCollector, CollectorConfig, ValidationError


@dataclass
class WikipediaConfig(CollectorConfig):
   """Wikipedia-specific configuration."""
   language: str = 'en'
   user_agent: str = 'GamingDatasetBot/1.0 (https://github.com/yourusername/gaming-dataset)'
   categories_per_batch: int = 10
   pages_per_category: int = 1000
   min_article_length: int = 1000  # caractères minimum
   include_plot_summaries: bool = False
   extract_infobox: bool = True
   

class WikipediaCollector(BaseCollector):
   """Collector for Wikipedia gaming content using official API."""
   
   # Catégories gaming prioritaires
   PRIORITY_CATEGORIES = [
       # High Priority - Recent & Featured
       'Category:Featured_video_game_articles',
       'Category:Good_video_game_articles', 
       'Category:2024_video_games',
       'Category:2023_video_games',
       'Category:2022_video_games',
       'Category:The_Game_Award_for_Game_of_the_Year_winners',
       
       # Game Mechanics & Design
       'Category:Video_game_gameplay',
       'Category:Video_game_mechanics',
       'Category:Video_game_terminology', 
       'Category:Video_game_genres',
       'Category:Video_game_design',
       
       # Franchises & Series
       'Category:Video_game_franchises',
       'Category:Video_game_sequels',
       'Category:Long-running_video_game_franchises',
       
       # Competitive & Esports
       'Category:Esports_games',
       'Category:Fighting_games',
       'Category:Multiplayer_online_battle_arena_games',
       'Category:First-person_shooters',
       'Category:Real-time_strategy_video_games',
       
       # Indies & Innovation
       'Category:Indie_video_games',
       'Category:IGF_Grand_Prize_winners',
       'Category:IndieCade_winners',
       
       # Platforms & Technical
       'Category:PlayStation_5_games',
       'Category:Xbox_Series_X_and_Series_S_games',
       'Category:Nintendo_Switch_games',
       'Category:Windows_games',
       'Category:Virtual_reality_games',
   ]
   
   def __init__(self, config: Optional[WikipediaConfig] = None):
       super().__init__(config or WikipediaConfig())
       self.config: WikipediaConfig = self.config
       
       # Initialize Wikipedia API
       self.wiki = wikipediaapi.Wikipedia(
           language=self.config.language,
           user_agent=self.config.user_agent
       )
       
       # Cache pour éviter les doublons
       self.processed_pages = set()
       self.category_cache = {}
       
       # Session aiohttp pour wptools
       self.session = None
       
   async def __aenter__(self):
       """Initialize async resources."""
       self.session = aiohttp.ClientSession()
       return self
       
   async def __aexit__(self, exc_type, exc_val, exc_tb):
       """Cleanup async resources."""
       if self.session:
           await self.session.close()
   
   async def extract(self) -> AsyncIterator[Dict]:
       """Extract gaming content from Wikipedia."""
       logger.info(f"Starting Wikipedia extraction with {len(self.PRIORITY_CATEGORIES)} categories")
       
       # Process each category
       for category_name in self.PRIORITY_CATEGORIES:
           logger.info(f"Processing category: {category_name}")
           
           try:
               async for document in self._process_category(category_name):
                   yield document
                   
           except Exception as e:
               logger.error(f"Error processing category {category_name}: {e}")
               continue
               
       logger.info(f"Wikipedia extraction complete. Processed {len(self.processed_pages)} unique pages")
   
   async def _process_category(self, category_name: str) -> AsyncIterator[Dict]:
       """Process all pages in a category."""
       category = self.wiki.page(category_name)
       
       if not category.exists():
           logger.warning(f"Category does not exist: {category_name}")
           return
           
       # Get all pages in category (not subcategories)
       pages_processed = 0
       
       for page_title in category.categorymembers.keys():
           # Skip subcategories
           if page_title.startswith('Category:'):
               continue
               
           # Skip if already processed
           if page_title in self.processed_pages:
               continue
               
           # Process page
           try:
               document = await self._process_page(page_title)
               if document:
                   self.processed_pages.add(page_title)
                   pages_processed += 1
                   yield document
                   
                   # Respect rate limits
                   if pages_processed % 10 == 0:
                       await asyncio.sleep(1)
                       
                   # Limit pages per category
                   if pages_processed >= self.config.pages_per_category:
                       break
                       
           except Exception as e:
               logger.error(f"Error processing page {page_title}: {e}")
               continue
   
   async def _process_page(self, page_title: str) -> Optional[Dict]:
       """Process a single Wikipedia page."""
       # Get page content
       page = self.wiki.page(page_title)
       
       if not page.exists():
           return None
           
       # Basic validation
       if len(page.text) < self.config.min_article_length:
           logger.debug(f"Skipping {page_title}: too short ({len(page.text)} chars)")
           return None
           
       # Extract content
       content = {
           'title': page.title,
           'text': page.text,
           'summary': page.summary,
           'url': page.fullurl,
           'categories': list(page.categories.keys()),
           'links': list(page.links.keys())[:100],  # Limit links
       }
       
       # Clean text
       content['text_clean'] = self._clean_wikipedia_text(content['text'])
       
       # Extract sections
       sections = self._extract_sections(page)
       if sections:
           content['sections'] = sections
           
       # Extract infobox if configured
       metadata = {
           'page_id': page.pageid,
           'categories': [cat.replace('Category:', '') for cat in content['categories']],
       }
       
       if self.config.extract_infobox:
           infobox_data = await self._extract_infobox(page_title)
           if infobox_data:
               metadata['game_info'] = self._parse_game_info(infobox_data)
       
       # Build document
       document = {
           'document_id': f"wiki_{page.pageid}_{int(time.time())}",
           'source': {
               'platform': 'wikipedia',
               'language': self.config.language,
               'url': page.fullurl,
               'crawled_at': datetime.utcnow().isoformat()
           },
           'content': content,
           'metadata': metadata
       }
       
       # Validate
       try:
           self._validate_document(document)
           return document
       except ValidationError as e:
           logger.debug(f"Document validation failed for {page_title}: {e}")
           return None
   
   def _clean_wikipedia_text(self, text: str) -> str:
       """Clean Wikipedia text from markup and boilerplate."""
       # Remove everything after common ending sections
       for section in ['See also', 'References', 'External links', 'Further reading', 'Notes', 'Bibliography']:
           pattern = f'\n{section}\n'
           if pattern in text:
               text = text.split(pattern)[0]
       
       # Remove Wikipedia-specific markup
       # Remove ref tags
       text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
       text = re.sub(r'<ref[^>]*/?>', '', text)
       
       # Remove citations [1], [2], etc.
       text = re.sub(r'\[\d+\]', '', text)
       text = re.sub(r'\[citation needed\]', '', text)
       
       # Remove edit links
       text = re.sub(r'\[edit\]', '', text)
       
       # Clean up whitespace
       text = re.sub(r'\n{3,}', '\n\n', text)
       text = re.sub(r' {2,}', ' ', text)
       
       # Remove file/image references
       text = re.sub(r'File:[^\n]+\n', '', text)
       text = re.sub(r'Image:[^\n]+\n', '', text)
       
       return text.strip()
   
   def _extract_sections(self, page) -> Dict[str, str]:
       """Extract important gaming-related sections."""
       sections = {}
       
       # Sections importantes pour le gaming
       important_sections = [
           'Gameplay', 'Plot', 'Development', 'Reception', 
           'Story', 'Characters', 'Setting', 'Combat',
           'Mechanics', 'Multiplayer', 'Game modes',
           'Synopsis', 'Premise', 'Critical reception',
           'Sales', 'Legacy', 'Soundtrack', 'Graphics'
       ]
       
       # Parse sections from page.sections
       # Note: wikipedia-api doesn't provide direct section access,
       # so we parse from text
       text_lines = page.text.split('\n')
       current_section = None
       section_content = []
       
       for line in text_lines:
           # Detect section headers (usually followed by multiple =)
           if line.strip() and not line.startswith(' '):
               # Simple heuristic: lines that might be headers
               potential_header = line.strip('= ')
               if potential_header in important_sections:
                   # Save previous section
                   if current_section and section_content:
                       sections[current_section] = '\n'.join(section_content).strip()
                   
                   current_section = potential_header.lower().replace(' ', '_')
                   section_content = []
               elif current_section:
                   section_content.append(line)
       
       # Don't forget last section
       if current_section and section_content:
           sections[current_section] = '\n'.join(section_content).strip()
       
       return sections
   
   async def _extract_infobox(self, page_title: str) -> Optional[Dict]:
       """Extract infobox data using wptools."""
       try:
           # Create wptools page
           page = wptools.page(page_title, silent=True)
           
           # Get parse data (includes infobox)
           # Note: wptools uses requests, so we run in executor
           loop = asyncio.get_event_loop()
           await loop.run_in_executor(None, page.get_parse)
           
           # Return infobox data
           return page.data.get('infobox', {})
           
       except Exception as e:
           logger.debug(f"Failed to extract infobox for {page_title}: {e}")
           return None
   
   def _parse_game_info(self, infobox: Dict) -> Dict:
       """Parse game information from infobox."""
       game_info = {}
       
       # Mapping des champs infobox vers nos champs
       field_mappings = {
           'developer': ['developer', 'developers'],
           'publisher': ['publisher', 'publishers'],
           'release_date': ['released', 'release date', 'release'],
           'genre': ['genre', 'genres'],
           'platforms': ['platform', 'platforms'],
           'director': ['director', 'directors'],
           'producer': ['producer', 'producers'],
           'designer': ['designer', 'designers'],
           'programmer': ['programmer', 'programmers'],
           'artist': ['artist', 'artists'],
           'writer': ['writer', 'writers'],
           'composer': ['composer', 'composers'],
           'series': ['series'],
           'engine': ['engine'],
           'modes': ['mode', 'modes'],
       }
       
       # Extract fields
       for our_field, infobox_fields in field_mappings.items():
           for field in infobox_fields:
               if field in infobox:
                   value = infobox[field]
                   # Clean up the value
                   if isinstance(value, str):
                       # Remove wiki markup
                       value = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', value)
                       value = re.sub(r'<[^>]+>', '', value)
                       value = value.strip()
                   
                   game_info[our_field] = value
                   break
       
       # Parse release date to extract year
       if 'release_date' in game_info:
           year_match = re.search(r'(19|20)\d{2}', game_info['release_date'])
           if year_match:
               game_info['release_year'] = int(year_match.group())
       
       return game_info
   
   def _validate_document(self, document: Dict):
       """Validate Wikipedia document meets quality standards."""
       # Check required fields
       required_fields = ['document_id', 'source', 'content', 'metadata']
       for field in required_fields:
           if field not in document:
               raise ValidationError(f"Missing required field: {field}")
       
       # Check content
       content = document['content']
       if len(content.get('text_clean', '')) < self.config.min_article_length:
           raise ValidationError(f"Content too short: {len(content.get('text_clean', ''))} chars")
       
       # Check if it's actually gaming-related
       gaming_indicators = [
           'game', 'gameplay', 'player', 'level', 'character',
           'multiplayer', 'single-player', 'console', 'PC',
           'PlayStation', 'Xbox', 'Nintendo', 'Steam'
       ]
       
       text_lower = content.get('text_clean', '').lower()
       if not any(indicator in text_lower for indicator in gaming_indicators):
           raise ValidationError("Content doesn't appear to be gaming-related")
       
       # Check for stub articles
       if 'stub' in text_lower and len(text_lower) < 2000:
           raise ValidationError("Stub article")


# Example usage
async def main():
   config = WikipediaConfig(
       language='en',
       categories_per_batch=5,
       pages_per_category=100,
       extract_infobox=True
   )
   
   async with WikipediaCollector(config) as collector:
       documents = []
       async for doc in collector.extract():
           documents.append(doc)
           print(f"Collected: {doc['content']['title']}")
           
           if len(documents) >= 10:  # Limit for example
               break
       
       print(f"\nCollected {len(documents)} documents")
       
       # Save example
       with open('wikipedia_sample.json', 'w', encoding='utf-8') as f:
           json.dump(documents[:3], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
   asyncio.run(main())