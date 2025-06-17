"""
Wikipedia collector for gaming-related articles - Enhanced version.
Extracts articles from gaming categories and processes them into the standard format.
"""

import asyncio
import re
from typing import Dict, List, Optional, AsyncIterator, Any, Set, Tuple
from datetime import datetime
import json

from loguru import logger
import aiohttp
from bs4 import BeautifulSoup

from collectors.base_collector import BaseCollector
from schemas import (
    Document, Content, Source, GameInfo, Classification, 
    DocumentMetadata, ContentType, Author
)
from utils import clean_text, extract_game_info_from_title


class WikipediaCollector(BaseCollector):
    """Enhanced collector for Wikipedia gaming articles using the MediaWiki API."""
    
    # Catégories prioritaires organisées par niveau
    PRIORITY_CATEGORIES = {
        'high': [
            # Articles de qualité Wikipedia
            'Category:Featured_video_game_articles',
            'Category:Good_video_game_articles',
            
            # Jeux récents et importants
            'Category:2024_video_games',
            'Category:2023_video_games', 
            'Category:2022_video_games',
            'Category:2021_video_games',
            'Category:The_Game_Award_for_Game_of_the_Year_winners',
            'Category:BAFTA_Games_Award_for_Best_Game_winners',
            'Category:Golden_Joystick_Award_for_Game_of_the_Year_winners',
            
            # Mécaniques et design
            'Category:Video_game_gameplay',
            'Category:Video_game_mechanics',
            'Category:Video_game_terminology',
            
            # Genres majeurs modernes
            'Category:Action_role-playing_video_games',
            'Category:Battle_royale_games',
            'Category:Soulslike_video_games',
            'Category:Roguelike_video_games',
            'Category:Metroidvania_games'
        ],
        'medium': [
            # Franchises et séries
            'Category:Video_game_franchises',
            'Category:Video_game_sequels',
            'Category:Video_game_remakes',
            
            # Esports et compétitif
            'Category:Esports_games',
            'Category:Fighting_games',
            'Category:First-person_shooters',
            'Category:Real-time_strategy_video_games',
            'Category:Multiplayer_online_battle_arena_games',
            
            # Indies et innovation
            'Category:Indie_video_games',
            'Category:IGF_Grand_Prize_winners',
            'Category:IndieCade_winners',
            
            # Plateformes majeures
            'Category:Steam_games',
            'Category:PlayStation_5_games',
            'Category:PlayStation_4_games',
            'Category:Xbox_Series_X_and_Series_S_games',
            'Category:Nintendo_Switch_games'
        ],
        'low': [
            # Contexte et culture
            'Category:Video_game_development',
            'Category:Video_game_developers',
            'Category:Video_game_publishers',
            'Category:Video_game_composers',
            'Category:Speedrunning',
            'Category:Video_game_modding',
            'Category:Virtual_reality_games',
            'Category:Mobile_games'
        ]
    }
    
    def __init__(self, start_year: int = 2000, priority_level: str = 'all'):
        """Initialize Wikipedia collector.
        
        Args:
            start_year: Earliest year to collect games from
            priority_level: 'high', 'medium', 'low', or 'all'
        """
        super().__init__({
            'rate_limit': 200,  # Wikipedia allows 200 req/s for registered apps
            'rate_period': 60,
            'headers': {
                'User-Agent': 'GamingDatasetBot/1.0 (https://github.com/yourusername/gaming-dataset) Python/3.10'
            }
        })
        
        self.start_year = start_year
        self.base_url = 'https://en.wikipedia.org/w/api.php'
        self.processed_pages: Set[str] = set()
        self.processed_titles: Set[str] = set()
        
        # Sélectionner les catégories selon le niveau de priorité
        if priority_level == 'all':
            self.categories_to_process = (
                self.PRIORITY_CATEGORIES['high'] +
                self.PRIORITY_CATEGORIES['medium'] +
                self.PRIORITY_CATEGORIES['low']
            )
        elif priority_level in self.PRIORITY_CATEGORIES:
            self.categories_to_process = self.PRIORITY_CATEGORIES[priority_level]
        else:
            raise ValueError(f"Invalid priority level: {priority_level}")
        
        logger.info(f"Initialized with {len(self.categories_to_process)} categories, priority: {priority_level}")
    
    async def extract(self) -> AsyncIterator[Dict[str, Any]]:
        """Extract gaming articles from Wikipedia categories."""
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint('wikipedia_collector.json')
        if checkpoint:
            self.processed_pages = set(checkpoint.get('processed_pages', []))
            self.processed_titles = set(checkpoint.get('processed_titles', []))
            logger.info(f"Resuming from checkpoint with {len(self.processed_pages)} processed pages")
        
        # Process each category
        for category in self.categories_to_process:
            logger.info(f"Processing category: {category}")
            
            try:
                # Get all pages in category with filtering
                all_pages = await self._get_all_category_pages(category)
                
                # Filter and score pages
                scored_pages = []
                for page in all_pages:
                    if page['pageid'] not in self.processed_pages and await self._should_process_article(page):
                        # Précharger un extrait pour le scoring
                        relevance_score = await self._calculate_initial_relevance(page)
                        if relevance_score > 0.3:  # Seuil minimal
                            scored_pages.append((relevance_score, page))
                
                # Trier par score de pertinence
                scored_pages.sort(key=lambda x: x[0], reverse=True)
                pages_to_process = [page for _, page in scored_pages]
                
                logger.info(f"Found {len(pages_to_process)} relevant pages in {category}")
                
                # Process in batches
                for batch in self.chunk_list(pages_to_process, 50):
                    documents = await self.process_batch(batch)
                    
                    for doc in documents:
                        if doc:
                            yield doc.to_dict()
                    
                    # Save checkpoint every batch
                    self.save_checkpoint({
                        'processed_pages': list(self.processed_pages),
                        'processed_titles': list(self.processed_titles),
                        'current_category': category
                    }, 'wikipedia_collector.json')
                    
            except Exception as e:
                logger.error(f"Error processing category {category}: {e}")
                continue
    
    async def _should_process_article(self, page_info: Dict[str, Any]) -> bool:
        """Filtrage intelligent des articles basé sur plusieurs critères."""
        title = page_info.get('title', '')
        
        # Skip patterns améliorés
        skip_patterns = [
            r'^List of',
            r'^Category:',
            r'^Template:',
            r'^Wikipedia:',
            r'^User:',
            r'^Talk:',
            r'disambiguation',
            r'\(film\)',
            r'\(TV series\)',
            r'\(soundtrack\)',
            r'\(company\)$',  # Companies at end of title
            r'in video gaming$',  # "2023 in video gaming" etc
            r'^Deaths in \d{4}',
            r'^Births in \d{4}',
            r'\(comics?\)',
            r'\(novel\)',
            r'\(book\)'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return False
        
        # Patterns à privilégier (override skip patterns)
        priority_patterns = [
            r'\(video game\)',
            r'\(\d{4} video game\)',
            r'gameplay',
            r'game mechanics',
            r'esports',
            r'speedrun'
        ]
        
        for pattern in priority_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return True
        
        return True
    
    async def _calculate_initial_relevance(self, page_info: Dict[str, Any]) -> float:
        """Calcule un score de pertinence initial basé sur les métadonnées."""
        score = 0.5  # Score de base
        title = page_info.get('title', '')
        
        # Bonus pour les patterns de jeux vidéo
        if re.search(r'\((?:video )?game\)', title, re.IGNORECASE):
            score += 0.3
        
        # Bonus pour l'année dans le titre (jeux récents)
        year_match = re.search(r'\b(20\d{2})\b', title)
        if year_match:
            year = int(year_match.group(1))
            if year >= 2020:
                score += 0.2
            elif year >= 2015:
                score += 0.1
        
        # Patterns de haute valeur dans le titre
        high_value_title_patterns = [
            'Game of the Year', 'GOTY', 'Remake', 'Remaster',
            'Definitive Edition', 'Enhanced Edition', 'Director\'s Cut'
        ]
        
        for pattern in high_value_title_patterns:
            if pattern.lower() in title.lower():
                score += 0.1
                break
        
        return min(score, 1.0)
    
    async def extract_item(self, page_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract content for a single Wikipedia page with enhanced metadata."""
        page_id = str(page_info['pageid'])
        page_title = page_info['title']
        
        # Skip if already processed
        if page_id in self.processed_pages or page_title in self.processed_titles:
            return None
        
        try:
            # Get page content with enhanced properties
            params = {
                'action': 'query',
                'prop': 'revisions|categories|info|extracts|pageprops',
                'pageids': page_id,
                'rvprop': 'content',
                'rvslots': 'main',
                'inprop': 'url',
                'exintro': '1',
                'explaintext': '1',
                'format': 'json',
                'formatversion': '2',
                'cllimit': 'max'  # Get all categories
            }
            
            data = await self.fetch_with_retry(self.base_url, params=params)
            
            if 'error' in data:
                logger.error(f"API error for page {page_title}: {data['error']}")
                return None
            
            pages = data.get('query', {}).get('pages', [])
            if not pages:
                return None
                
            page = pages[0]
            
            # Get full content
            content = ''
            if 'revisions' in page and page['revisions']:
                content = page['revisions'][0]['slots']['main'].get('content', '')
            
            # Skip if too short
            if len(content) < 500:
                logger.debug(f"Skipping {page_title} - content too short")
                return None
            
            # Extract enhanced metadata
            enhanced_metadata = self._extract_enhanced_metadata(content, page)
            
            # Calculate detailed relevance score
            relevance_score = self._calculate_relevance_score(page_info, content, enhanced_metadata)
            
            # Skip if relevance too low
            if relevance_score < 0.4:
                logger.debug(f"Skipping {page_title} - low relevance score: {relevance_score:.2f}")
                return None
            
            # Mark as processed
            self.processed_pages.add(page_id)
            self.processed_titles.add(page_title)
            
            return {
                'page_id': page_id,
                'title': page_title,
                'content': content,
                'categories': [cat['title'] for cat in page.get('categories', [])],
                'url': page.get('fullurl', f"https://en.wikipedia.org/?curid={page_id}"),
                'extract': page.get('extract', ''),
                'timestamp': page_info.get('timestamp', datetime.now().isoformat()),
                'enhanced_metadata': enhanced_metadata,
                'relevance_score': relevance_score
            }
            
        except Exception as e:
            logger.error(f"Error extracting page {page_title}: {e}")
            self.error_count += 1
            return None
    
    def _extract_enhanced_metadata(self, content: str, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extraction enrichie des métadonnées avec plus de contexte."""
        metadata = {}
        
        # 1. Extraire l'infobox
        infobox_data = self._extract_infobox_data(content)
        metadata['infobox'] = infobox_data
        
        # 2. Analyser les catégories
        categories = page_data.get('categories', [])
        metadata.update(self._analyze_categories(categories))
        
        # 3. Extraire les sections présentes
        sections = re.findall(r'^==+\s*([^=]+)\s*==+', content, re.MULTILINE)
        metadata['sections'] = [s.strip() for s in sections]
        metadata['has_gameplay_section'] = any('gameplay' in s.lower() for s in sections)
        metadata['has_development_section'] = any('development' in s.lower() for s in sections)
        metadata['has_reception_section'] = any('reception' in s.lower() for s in sections)
        
        # 4. Détecter les patterns de contenu
        metadata.update(self._detect_content_patterns(content))
        
        # 5. Statistiques du contenu
        metadata['word_count'] = len(content.split())
        metadata['section_count'] = len(sections)
        metadata['has_infobox'] = bool(infobox_data)
        
        # 6. Qualité de l'article Wikipedia
        metadata['is_featured'] = '{{Featured article}}' in content[:1000]
        metadata['is_good_article'] = '{{Good article}}' in content[:1000]
        
        return metadata
    
    def _extract_infobox_data(self, content: str) -> Dict[str, Any]:
        """Extraction améliorée des données depuis l'infobox Wikipedia."""
        infobox_data = {}
        
        # Pattern pour l'infobox avec gestion des nested templates
        infobox_pattern = r'\{\{Infobox video game(.*?)\n\}\}'
        match = re.search(infobox_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            infobox_content = match.group(1)
            
            # Patterns pour extraire les champs
            fields = {
                'developer': r'\|\s*developer\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'publisher': r'\|\s*publisher\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'director': r'\|\s*director\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'producer': r'\|\s*producer\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'designer': r'\|\s*designer\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'programmer': r'\|\s*programmer\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'artist': r'\|\s*artist\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'writer': r'\|\s*writer\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'composer': r'\|\s*composer\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'series': r'\|\s*series\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'engine': r'\|\s*engine\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'platforms': r'\|\s*platform\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'released': r'\|\s*released\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'genre': r'\|\s*genre\s*=\s*([^\|]+?)(?=\n\||\n\})',
                'modes': r'\|\s*modes\s*=\s*([^\|]+?)(?=\n\||\n\})',
            }
            
            for field, pattern in fields.items():
                match = re.search(pattern, infobox_content, re.DOTALL)
                if match:
                    # Nettoyer les liens wiki et le formatage
                    value = match.group(1)
                    value = self._clean_wiki_markup(value)
                    
                    if value:
                        infobox_data[field] = value
        
        return infobox_data
    
    def _clean_wiki_markup(self, text: str) -> str:
        """Nettoie le markup Wikipedia d'un texte."""
        # Liens internes [[Link|Display]] -> Display, [[Link]] -> Link
        text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
        # Templates simples
        text = re.sub(r'\{\{(?:nowrap|small|nobr)\|([^}]+)\}\}', r'\1', text)
        # Retirer les autres templates
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        # HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Références
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/?>', '', text)
        # Multiple espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _analyze_categories(self, categories: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyse les catégories pour extraire genres, plateformes, etc."""
        result = {
            'enhanced_genres': set(),
            'enhanced_tags': set(),
            'platforms_detected': set(),
            'release_years': set()
        }
        
        for category in categories:
            cat_title = category.get('title', '').replace('Category:', '')
            cat_lower = cat_title.lower()
            
            # Genres
            genre_mapping = {
                'action': ['action'],
                'adventure': ['adventure'],
                'rpg': ['role-playing', 'rpg'],
                'strategy': ['strategy', 'rts', 'turn-based'],
                'simulation': ['simulation', 'sim'],
                'sports': ['sports', 'racing', 'football', 'basketball'],
                'puzzle': ['puzzle'],
                'platformer': ['platform', 'platformer'],
                'shooter': ['shooter', 'fps', 'third-person shooter'],
                'fighting': ['fighting'],
                'horror': ['horror', 'survival horror'],
                'stealth': ['stealth'],
                'mmorpg': ['massively multiplayer'],
                'battle-royale': ['battle royale'],
                'roguelike': ['roguelike', 'roguelite'],
                'metroidvania': ['metroidvania'],
                'soulslike': ['soulslike', 'souls-like']
            }
            
            for genre, keywords in genre_mapping.items():
                if any(kw in cat_lower for kw in keywords):
                    result['enhanced_genres'].add(genre)
            
            # Tags spéciaux
            tag_mapping = {
                'multiplayer': ['multiplayer', 'online'],
                'single-player': ['single-player', 'single player'],
                'indie': ['indie', 'independent'],
                'free-to-play': ['free-to-play', 'free to play'],
                'open-world': ['open world', 'open-world'],
                'vr': ['virtual reality', 'vr games'],
                'mobile': ['ios games', 'android games', 'mobile games'],
                'esports': ['esports'],
                'early-access': ['early access'],
                'remake': ['video game remakes'],
                'remaster': ['video game remasters']
            }
            
            for tag, keywords in tag_mapping.items():
                if any(kw in cat_lower for kw in keywords):
                    result['enhanced_tags'].add(tag)
            
            # Plateformes
            platform_mapping = {
                'PC': ['windows games', 'linux games', 'macos games', 'dos games', 'steam games'],
                'PlayStation': ['playstation', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5'],
                'Xbox': ['xbox', 'xbox 360', 'xbox one', 'xbox series'],
                'Nintendo': ['nintendo', 'switch games', 'wii', '3ds', 'game boy'],
                'Mobile': ['ios games', 'android games', 'mobile games']
            }
            
            for platform, keywords in platform_mapping.items():
                if any(kw in cat_lower for kw in keywords):
                    result['platforms_detected'].add(platform)
            
            # Années
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', cat_title)
            if year_match:
                result['release_years'].add(int(year_match.group(1)))
        
        # Convertir les sets en listes
        for key in ['enhanced_genres', 'enhanced_tags', 'platforms_detected', 'release_years']:
            result[key] = sorted(list(result[key]))
        
        return result
    
    def _detect_content_patterns(self, content: str) -> Dict[str, Any]:
        """Détecte des patterns de contenu spécifiques."""
        content_lower = content.lower()
        
        patterns = {
            'has_multiplayer': bool(re.search(r'\bmultiplayer\b|\bonline\s+(?:play|mode|multiplayer)', content_lower)),
            'has_dlc': bool(re.search(r'\bDLC\b|downloadable content|expansion pack', content_lower)),
            'has_microtransactions': bool(re.search(r'microtransaction|in-app purchase|loot box', content_lower)),
            'is_free_to_play': bool(re.search(r'free-to-play|free to play|f2p', content_lower)),
            'has_season_pass': bool(re.search(r'season pass|battle pass', content_lower)),
            'has_modding': bool(re.search(r'\bmod(?:ding|s)?\b|modification|workshop', content_lower)),
            'has_speedrun': bool(re.search(r'speedrun|speed run', content_lower)),
            'has_esports': bool(re.search(r'esports|e-sports|tournament|competitive', content_lower)),
            'development_time_mentioned': bool(re.search(r'development (?:began|started|took)', content_lower)),
            'budget_mentioned': bool(re.search(r'budget|cost.*develop|development cost', content_lower)),
            'sales_mentioned': bool(re.search(r'sold.*(?:copies|units)|sales figure', content_lower)),
            'awards_mentioned': bool(re.search(r'award|nominated|won|winner', content_lower))
        }
        
        return patterns
    
    def _calculate_relevance_score(self, page_info: Dict[str, Any], 
                                  content: str, metadata: Dict[str, Any]) -> float:
        """Calcule un score de pertinence détaillé pour prioriser les articles."""
        score = 0.0
        
        # 1. Score basé sur la qualité Wikipedia (0-0.3)
        if metadata.get('is_featured'):
            score += 0.3
        elif metadata.get('is_good_article'):
            score += 0.2
        elif metadata.get('word_count', 0) > 5000:
            score += 0.1
        
        # 2. Score basé sur la complétude (0-0.2)
        important_sections = ['gameplay', 'development', 'reception']
        sections_present = sum(1 for s in important_sections if metadata.get(f'has_{s}_section'))
        score += (sections_present / len(important_sections)) * 0.2
        
        # 3. Score basé sur les métadonnées (0-0.2)
        if metadata.get('has_infobox'):
            score += 0.1
        if len(metadata.get('enhanced_genres', [])) >= 2:
            score += 0.05
        if len(metadata.get('platforms_detected', [])) >= 2:
            score += 0.05
        
        # 4. Score basé sur l'importance du jeu (0-0.2)
        # Jeux récents
        years = metadata.get('release_years', [])
        if years and max(years) >= 2020:
            score += 0.1
        elif years and max(years) >= 2015:
            score += 0.05
        
        # Franchise
        if metadata.get('infobox', {}).get('series'):
            score += 0.05
        
        # Popularité (esports, awards, etc.)
        if metadata.get('has_esports'):
            score += 0.05
        if metadata.get('awards_mentioned'):
            score += 0.05
        
        # 5. Score basé sur le contenu gaming (0-0.1)
        gaming_patterns = [
            'has_multiplayer', 'has_modding', 'has_speedrun',
            'development_time_mentioned', 'sales_mentioned'
        ]
        gaming_score = sum(1 for p in gaming_patterns if metadata.get(p, False))
        score += (gaming_score / len(gaming_patterns)) * 0.1
        
        return min(score, 1.0)
        
    async def extract(self) -> AsyncIterator[Dict[str, Any]]:
        """Extract gaming articles from Wikipedia categories."""
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint('wikipedia_collector.json')
        if checkpoint:
            self.processed_pages = set(checkpoint.get('processed_pages', []))
            self.processed_titles = set(checkpoint.get('processed_titles', []))
            logger.info(f"Resuming from checkpoint with {len(self.processed_pages)} processed pages")
        
        # Process each category
        for category in self.CATEGORIES:
            logger.info(f"Processing category: {category}")
            
            try:
                # Get all pages in category (with pagination)
                all_pages = await self._get_all_category_pages(category)
                logger.info(f"Found {len(all_pages)} total pages in {category}")
                
                # Filter out already processed
                new_pages = [p for p in all_pages if p['pageid'] not in self.processed_pages]
                logger.info(f"Processing {len(new_pages)} new pages from {category}")
                
                # Process in batches
                for batch in self.chunk_list(new_pages, 50):
                    documents = await self.process_batch(batch)
                    
                    for doc in documents:
                        if doc:  # Check if document is not None
                            yield doc.to_dict()
                    
                    # Save checkpoint every batch
                    self.save_checkpoint({
                        'processed_pages': list(self.processed_pages),
                        'processed_titles': list(self.processed_titles),
                        'current_category': category
                    }, 'wikipedia_collector.json')
                    
            except Exception as e:
                logger.error(f"Error processing category {category}: {e}")
                continue
    
    async def _get_all_category_pages(self, category: str) -> List[Dict[str, Any]]:
        """Get all pages in a category with pagination."""
        all_pages = []
        continue_token = None
        
        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmlimit': 500,  # Maximum allowed
                'cmtype': 'page',  # Only get pages, not subcategories
                'format': 'json',
                'cmprop': 'ids|title|timestamp'
            }
            
            if continue_token:
                params['cmcontinue'] = continue_token
            
            try:
                data = await self.fetch_with_retry(self.base_url, params=params)
                
                if 'error' in data:
                    logger.error(f"API error for category {category}: {data['error']}")
                    break
                
                pages = data.get('query', {}).get('categorymembers', [])
                
                # Filter by year if in title
                for page in pages:
                    title = page.get('title', '')
                    
                    # Skip if already processed by title (handles redirects)
                    if title in self.processed_titles:
                        continue
                    
                    # Extract year from title if present
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    if year_match:
                        year = int(year_match.group())
                        if year < self.start_year:
                            continue
                    
                    # Skip list pages and disambiguation pages
                    if any(skip in title.lower() for skip in ['list of', 'disambiguation', 'category:']):
                        continue
                    
                    all_pages.append(page)
                
                # Check for continuation
                if 'continue' in data:
                    continue_token = data['continue'].get('cmcontinue')
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching category {category}: {e}")
                break
        
        return all_pages
    
    async def extract_item(self, page_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract content for a single Wikipedia page."""
        page_id = str(page_info['pageid'])
        page_title = page_info['title']
        
        # Skip if already processed
        if page_id in self.processed_pages or page_title in self.processed_titles:
            return None
        
        try:
            # Get page content
            params = {
                'action': 'query',
                'prop': 'revisions|categories|info|extracts',
                'pageids': page_id,
                'rvprop': 'content',
                'rvslots': 'main',
                'inprop': 'url',
                'exintro': 1,  # Get introduction
                'explaintext': 1,  # Plain text
                'format': 'json',
                'formatversion': 2
            }
            
            data = await self.fetch_with_retry(self.base_url, params=params)
            
            if 'error' in data:
                logger.error(f"API error for page {page_title}: {data['error']}")
                return None
            
            pages = data.get('query', {}).get('pages', [])
            if not pages:
                return None
                
            page = pages[0]
            
            # Get full content
            content = ''
            if 'revisions' in page and page['revisions']:
                content = page['revisions'][0]['slots']['main'].get('content', '')
            
            # Skip if too short
            if len(content) < 500:
                logger.debug(f"Skipping {page_title} - content too short")
                return None
            
            # Mark as processed
            self.processed_pages.add(page_id)
            self.processed_titles.add(page_title)
            
            return {
                'page_id': page_id,
                'title': page_title,
                'content': content,
                'categories': [cat['title'] for cat in page.get('categories', [])],
                'url': page.get('fullurl', f"https://en.wikipedia.org/?curid={page_id}"),
                'extract': page.get('extract', ''),  # Introduction text
                'timestamp': page_info.get('timestamp', datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error extracting page {page_title}: {e}")
            self.error_count += 1
            return None
    
    def parse(self, raw_data: Optional[Dict[str, Any]]) -> Optional[Document]:
        """Parse Wikipedia page into Document format."""
        if not raw_data:
            return None
            
        try:
            # Clean wiki markup
            text_clean = self._clean_wiki_text(raw_data['content'])
            
            # Skip if cleaned text is too short
            if len(text_clean) < 300:
                return None
            
            # Extract game info
            game_info = self._extract_game_info(
                raw_data['title'], 
                text_clean,
                raw_data.get('extract', '')
            )
            
            # Determine content type
            content_type = self._determine_content_type(
                raw_data['title'],
                raw_data['categories']
            )
            
            # Extract metadata
            genres = self._extract_genres(raw_data['categories'], text_clean)
            platforms = self._extract_platforms(text_clean)
            tags = self._extract_tags(raw_data['categories'], text_clean)
            
            # Create document
            document = Document(
                document_id=self.generate_document_id(raw_data['url']),
                source=Source(
                    platform='wikipedia',
                    url=raw_data['url'],
                    api_endpoint='MediaWiki API'
                ),
                content=Content(
                    title=raw_data['title'],
                    text=raw_data['content'],
                    text_clean=text_clean
                ),
                metadata=DocumentMetadata(
                    game=game_info,
                    classification=Classification(
                        content_type=content_type,
                        genres=genres,
                        platforms=platforms,
                        tags=tags,
                        is_tutorial='guide' in raw_data['title'].lower() or 'walkthrough' in raw_data['title'].lower(),
                        is_review='review' in ' '.join(raw_data['categories']).lower(),
                        is_news=False  # Wikipedia is not news
                    ),
                    author=Author(
                        name='Wikipedia Contributors',
                        platform='wikipedia',
                        verified=True
                    )
                )
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error parsing Wikipedia page {raw_data.get('title', 'Unknown')}: {e}")
            return None
    
    def _clean_wiki_text(self, wiki_markup: str) -> str:
        """Clean Wikipedia markup to plain text."""
        # Remove templates in double braces
        text = re.sub(r'\{\{[^}]+\}\}', '', wiki_markup)
        
        # Remove references
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/?>', '', text)
        
        # Remove comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove tables
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        
        # Convert wiki links to plain text
        # [[Link|Display]] -> Display
        # [[Link]] -> Link
        text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
        
        # Remove external links but keep text
        text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        text = re.sub(r'\[https?://[^\s\]]+\]', '', text)
        
        # Remove images/files
        text = re.sub(r'\[\[(?:File|Image):[^\]]+\]\]', '', text)
        
        # Convert headers
        text = re.sub(r'^={2,6}\s*(.+?)\s*={2,6}$', r'\n\1\n', text, flags=re.MULTILINE)
        
        # Remove categories at bottom
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
        
        # Clean up HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t', ' ', text)
        
        # Remove common Wikipedia boilerplate
        text = self._remove_wikipedia_boilerplate(text)
        
        return text.strip()
    
    def _remove_wikipedia_boilerplate(self, text: str) -> str:
        """Remove common Wikipedia boilerplate text."""
        # Common patterns to remove
        boilerplate_patterns = [
            r'From Wikipedia, the free encyclopedia',
            r'Jump to navigation.*?Jump to search',
            r'This article is about.*?\. For other uses, see',
            r'Not to be confused with',
            r'\[edit\]',
            r'Retrieved from "https://.*?"',
            r'Categories:.*$',
            r'Hidden categories:.*$'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def _extract_game_info(self, title: str, content: str, extract: str) -> GameInfo:
        """Extract game information from title and content."""
        # Clean game title
        game_title = title
        for suffix in [' (video game)', ' (game)', ' (series)', ' (franchise)']:
            game_title = game_title.replace(suffix, '')
        
        # Initialize game info
        game_info = GameInfo(title=game_title)
        
        # Use extract for better accuracy (it's the introduction)
        search_text = extract if extract else content[:2000]
        
        # Extract release date - multiple patterns
        release_patterns = [
            #r'released?(?:\s+on)',
            r'released?(?:\s+on)?\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'released?(?:\s+in)?\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'released?(?:\s+in)?\s+(\d{4})',
            r'launch(?:ed)?\s+(?:on|in)\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'came out\s+(?:on|in)\s+(\w+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in release_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                game_info.release_date = match.group(1).strip()
                break
        
        # Extract developer
        dev_patterns = [
            r'developed\s+by\s+([A-Z][^.,:;]+?)(?:\s+and\s+published|\s+for|\.|,|;|:|\s+is\s)',
            r'developer\s+([A-Z][^.,:;]+?)(?:\.|,|;|:)',
            r'created\s+by\s+([A-Z][^.,:;]+?)(?:\.|,|;|:)',
            r'made\s+by\s+([A-Z][^.,:;]+?)(?:\.|,|;|:)'
        ]
        
        for pattern in dev_patterns:
            match = re.search(pattern, search_text)
            if match:
                game_info.developer = match.group(1).strip()
                break
        
        # Extract publisher
        pub_patterns = [
            r'published\s+by\s+([A-Z][^.,:;]+?)(?:\.|,|;|:|\s+for\s)',
            r'publisher\s+([A-Z][^.,:;]+?)(?:\.|,|;|:)',
            r'distributed\s+by\s+([A-Z][^.,:;]+?)(?:\.|,|;|:)'
        ]
        
        for pattern in pub_patterns:
            match = re.search(pattern, search_text)
            if match:
                game_info.publisher = match.group(1).strip()
                break
        
        # Extract genres from content
        genre_patterns = [
            r'is\s+an?\s+([^.]+?)\s+(?:video\s+)?game',
            r'is\s+an?\s+([^.]+?)\s+developed',
            r'([^.]+?)\s+video\s+game\s+(?:developed|created|published)',
            r'genre[s]?\s*:\s*([^.;\n]+)'
        ]
        
        for pattern in genre_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                genre_text = match.group(1).lower()
                # Extract individual genres
                genre_keywords = [
                    'action', 'adventure', 'role-playing', 'rpg', 'strategy',
                    'simulation', 'sports', 'racing', 'puzzle', 'platformer',
                    'shooter', 'fighting', 'survival', 'horror', 'stealth',
                    'sandbox', 'mmorpg', 'moba', 'battle royale', 'roguelike',
                    'tactical', 'turn-based', 'real-time', 'multiplayer'
                ]
                
                for keyword in genre_keywords:
                    if keyword in genre_text:
                        if keyword == 'role-playing' or keyword == 'rpg':
                            game_info.genres.append('RPG')
                        else:
                            game_info.genres.append(keyword.title())
                
                if game_info.genres:
                    break
        
        # Deduplicate genres
        game_info.genres = list(set(game_info.genres))
        
        return game_info
    
    def _determine_content_type(self, title: str, categories: List[str]) -> ContentType:
        """Determine content type from title and categories."""
        title_lower = title.lower()
        categories_str = ' '.join(categories).lower()
        
        # Check title patterns
        if any(x in title_lower for x in ['guide', 'walkthrough', 'tutorial']):
            return ContentType.GUIDE
        elif 'review' in title_lower:
            return ContentType.REVIEW
        elif any(x in title_lower for x in ['list of', 'comparison', 'timeline']):
            return ContentType.WIKI
        
        # Check categories
        if any(x in categories_str for x in ['guides', 'walkthroughs', 'tutorials']):
            return ContentType.GUIDE
        elif any(x in categories_str for x in ['reviews', 'reception']):
            return ContentType.REVIEW
        elif any(x in categories_str for x in ['stubs', 'lists']):
            return ContentType.WIKI
        elif any(x in categories_str for x in ['companies', 'developers', 'publishers', 'people']):
            return ContentType.ENCYCLOPEDIA
        
        # Default to encyclopedia for game articles
        return ContentType.ENCYCLOPEDIA
    
    def _extract_genres(self, categories: List[str], content: str) -> List[str]:
        """Extract game genres from categories and content."""
        genres = set()
        
        # Genre mapping
        genre_map = {
            'action': ['action', 'beat \'em up', 'hack and slash'],
            'adventure': ['adventure', 'point-and-click'],
            'rpg': ['role-playing', 'rpg', 'jrpg', 'wrpg', 'action rpg'],
            'strategy': ['strategy', 'rts', 'real-time strategy', 'turn-based strategy', 'tactical'],
            'simulation': ['simulation', 'sim', 'life simulation', 'business simulation'],
            'sports': ['sports', 'racing', 'football', 'basketball', 'soccer'],
            'puzzle': ['puzzle', 'match-3', 'tetris'],
            'platformer': ['platform', 'platformer', 'metroidvania'],
            'shooter': ['shooter', 'fps', 'first-person shooter', 'third-person shooter', 'shoot \'em up'],
            'fighting': ['fighting', 'fighter', 'beat \'em up'],
            'survival': ['survival', 'survival horror'],
            'horror': ['horror', 'survival horror'],
            'stealth': ['stealth'],
            'sandbox': ['sandbox', 'open world'],
            'mmorpg': ['mmorpg', 'massively multiplayer'],
            'moba': ['moba', 'multiplayer online battle arena'],
            'battle royale': ['battle royale'],
            'roguelike': ['roguelike', 'roguelite'],
            'rhythm': ['rhythm', 'music'],
            'visual novel': ['visual novel'],
            'indie': ['indie', 'independent']
        }
        
        # Check categories
        categories_lower = ' '.join(categories).lower()
        for genre, keywords in genre_map.items():
            if any(keyword in categories_lower for keyword in keywords):
                genres.add(genre.replace('_', ' ').title())
        
        # Check content (first 1000 chars for efficiency)
        content_lower = content[:1000].lower()
        for genre, keywords in genre_map.items():
            if any(keyword in content_lower for keyword in keywords):
                genres.add(genre.replace('_', ' ').title())
        
        return sorted(list(genres))
    
    def _extract_platforms(self, content: str) -> List[str]:
        """Extract gaming platforms from content."""
        platforms = set()
        
        # Platform patterns with their canonical names
        platform_patterns = {
            'PC': [
                r'\b(?:PC|personal computer|Windows|Microsoft Windows|MS Windows)\b',
                r'\b(?:Steam|Epic Games Store|GOG|Origin|Uplay)\b',
                r'\b(?:Linux|Mac|macOS|OS X)\b'
            ],
            'PlayStation 5': [r'\b(?:PlayStation 5|PS5)\b'],
            'PlayStation 4': [r'\b(?:PlayStation 4|PS4)\b'],
            'PlayStation 3': [r'\b(?:PlayStation 3|PS3)\b'],
            'Xbox Series X/S': [r'\b(?:Xbox Series [XS]|Xbox Series|Series X|Series S)\b'],
            'Xbox One': [r'\bXbox One\b'],
            'Xbox 360': [r'\bXbox 360\b'],
            'Nintendo Switch': [r'\b(?:Nintendo Switch|Switch)\b'],
            'Nintendo Wii U': [r'\b(?:Wii U|WiiU)\b'],
            'Nintendo Wii': [r'\b(?:Wii)\b(?! U)'],
            'Nintendo 3DS': [r'\b(?:Nintendo 3DS|3DS|2DS)\b'],
            'Mobile': [
                r'\b(?:iOS|iPhone|iPad|Apple App Store)\b',
                r'\b(?:Android|Google Play|Play Store)\b',
                r'\b(?:mobile|smartphone|tablet)\b'
            ],
            'VR': [
                r'\b(?:VR|virtual reality|Oculus|Quest|Vive|PSVR|PlayStation VR)\b',
                r'\b(?:SteamVR|Windows Mixed Reality)\b'
            ]
        }
        
        # Search content for platforms
        for platform, patterns in platform_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    platforms.add(platform)
                    break
        
        return sorted(list(platforms))
    
    def _extract_tags(self, categories: List[str], content: str) -> List[str]:
        """Extract relevant tags from categories and content."""
        tags = set()
        
        # Tag patterns
        tag_patterns = {
            'multiplayer': r'\b(?:multiplayer|multi-player|online multiplayer|local multiplayer)\b',
            'single-player': r'\b(?:single-player|singleplayer|solo|campaign)\b',
            'co-op': r'\b(?:co-op|coop|cooperative|co-operative)\b',
            'online': r'\b(?:online|network|internet)\b',
            'offline': r'\b(?:offline|local)\b',
            'free-to-play': r'\b(?:free-to-play|free to play|f2p|freemium)\b',
            'early-access': r'\b(?:early access|beta|alpha)\b',
            'indie': r'\b(?:indie|independent)\b',
            'aaa': r'\b(?:AAA|triple-A|blockbuster)\b',
            'remake': r'\b(?:remake|remaster|reboot|reimagining)\b',
            'sequel': r'\b(?:sequel|prequel|\b2\b|\bII\b|\bIII\b)\b',
            'dlc': r'\b(?:DLC|downloadable content|expansion|add-on)\b',
            'mod-support': r'\b(?:mod|modding|modification|workshop)\b',
            'competitive': r'\b(?:competitive|esports|e-sports|tournament)\b',
            'story-driven': r'\b(?:story-driven|narrative|story-focused|plot)\b',
            'open-world': r'\b(?:open world|open-world|sandbox|free roam)\b',
            'linear': r'\b(?:linear|corridor|guided)\b',
            'procedural': r'\b(?:procedural|randomly generated|roguelike)\b',
            'retro': r'\b(?:retro|classic|old-school|8-bit|16-bit|pixel)\b',
            'vr-compatible': r'\b(?:VR compatible|VR support|virtual reality)\b',
            'controller-support': r'\b(?:controller support|gamepad|joystick)\b'
        }
        
        # Check both categories and content
        combined_text = ' '.join(categories) + ' ' + content[:2000]
        
        for tag, pattern in tag_patterns.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                tags.add(tag)
        
        # Add special tags based on categories
        categories_lower = ' '.join(categories).lower()
        if 'esports' in categories_lower:
            tags.add('esports')
        if 'free-to-play' in categories_lower:
            tags.add('free-to-play')
        if 'early access' in categories_lower:
            tags.add('early-access')
        if 'virtual reality' in categories_lower:
            tags.add('vr')
        
        return sorted(list(tags))
    
    async def process_item(self, page_info: Dict[str, Any]) -> Optional[Document]:
        """Process a single Wikipedia page."""
        try:
            # Extract raw data
            raw_data = await self.extract_item(page_info)
            if not raw_data:
                return None
            
            # Parse into Document
            document = self.parse(raw_data)
            
            if document and self.validate(document):
                self.collected_count += 1
                return document
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing page {page_info.get('title', 'Unknown')}: {e}")
            self.error_count += 1
            return None