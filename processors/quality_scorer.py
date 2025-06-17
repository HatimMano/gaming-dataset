"""
Enhanced quality scoring system for gaming dataset with domain-specific metrics.
"""

import re
import math
from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import spacy
from loguru import logger

from utils import load_gaming_dictionary


@dataclass
class QualityWeights:
    """Weights optimized for gaming content."""
    gaming_density: float = 0.30      # Réduit de 0.35
    content_depth: float = 0.25       # NOUVEAU - profondeur du contenu
    structure: float = 0.15           # Réduit de 0.20
    uniqueness: float = 0.15          # Réduit de 0.20
    length: float = 0.10              # Réduit de 0.15
    freshness: float = 0.05           # Réduit de 0.10


class EnhancedQualityScorer:
    """Enhanced scorer with gaming-specific metrics."""
    
    def __init__(self, weights: Optional[QualityWeights] = None):
        self.weights = weights or QualityWeights()
        self.gaming_terms = load_gaming_dictionary()
        
        # Charger les vocabulaires spécialisés
        self._load_specialized_vocabularies()
        
        # Patterns de contenu de haute valeur
        self._compile_high_value_patterns()
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            self.use_spacy = False
    
    def _load_specialized_vocabularies(self):
        """Load specialized gaming vocabularies with categories."""
        
        # Mécaniques de gameplay (très haute valeur)
        self.gameplay_mechanics = {
            'combo', 'frame-data', 'hitbox', 'hurtbox', 'i-frames', 'invincibility-frames',
            'dodge-roll', 'parry', 'counter', 'juggle', 'cancel', 'buffer',
            'cooldown', 'proc', 'aggro', 'kiting', 'tanking', 'dps',
            'crowd-control', 'cc', 'stun', 'root', 'silence', 'debuff', 'buff',
            'crit', 'critical-hit', 'headshot', 'backstab', 'stealth',
            'respawn', 'checkpoint', 'save-state', 'permadeath', 'lives',
            'health-bar', 'mana-bar', 'stamina-bar', 'rage-meter', 'super-meter',
            'level-up', 'exp', 'xp', 'skill-tree', 'talent-tree', 'perks',
            'loot', 'drops', 'rng', 'gacha', 'crafting', 'enchanting',
            'fast-travel', 'waypoint', 'minimap', 'fog-of-war', 'line-of-sight'
        }
        
        # Termes techniques spécifiques
        self.technical_terms = {
            'fps', 'frames-per-second', 'frame-rate', 'tick-rate', 'ping', 'latency',
            'lag', 'rubber-banding', 'netcode', 'rollback', 'p2p', 'dedicated-server',
            'anti-aliasing', 'vsync', 'ray-tracing', 'dlss', 'fsr', 'tessellation',
            'lod', 'draw-distance', 'fov', 'field-of-view', 'resolution',
            'hitmarker', 'aim-assist', 'bullet-drop', 'hitscan', 'projectile',
            'physics-engine', 'ragdoll', 'collision-detection', 'clipping'
        }
        
        # Termes d'esports et compétitif
        self.esports_terms = {
            'meta', 'tier-list', 'viable', 'op', 'overpowered', 'nerf', 'buff',
            'patch', 'balance', 'matchmaking', 'mmr', 'elo', 'rank', 'ladder',
            'scrim', 'pug', 'tryhard', 'casual', 'ranked', 'placement',
            'main', 'one-trick', 'flex', 'carry', 'support', 'feeder',
            'tilt', 'toxic', 'gg', 'glhf', 'wp', 'throw', 'clutch',
            'ace', 'teamwipe', 'comeback', 'snowball', 'turtle',
            'push', 'rotate', 'flank', 'peek', 'camp', 'rush'
        }
        
        # Genres spécifiques et sous-genres
        self.genre_specific_terms = {
            'roguelike': ['permadeath', 'procedural', 'run', 'meta-progression'],
            'moba': ['lane', 'jungle', 'gank', 'last-hit', 'deny', 'ward'],
            'battle-royale': ['circle', 'zone', 'hot-drop', 'third-party', 'heal-off'],
            'mmorpg': ['raid', 'dungeon', 'guild', 'tank', 'healer', 'dps', 'mob'],
            'fighting': ['combo', 'frame-data', 'mixup', 'footsies', 'neutral'],
            'fps': ['frag', 'camping', 'spawn-kill', 'wallbang', 'prefire'],
            'rts': ['micro', 'macro', 'apm', 'build-order', 'rush', 'turtle'],
            'souls-like': ['bonfire', 'estus', 'souls', 'hollow', 'invade']
        }
        
        # Phrases complexes (2-3 mots) de haute valeur
        self.high_value_phrases = [
            'game feel', 'skill ceiling', 'skill floor', 'power creep',
            'quality of life', 'day one patch', 'review bombing', 'early access',
            'season pass', 'battle pass', 'loot box', 'pay to win',
            'free to play', 'live service', 'always online', 'split screen',
            'couch co-op', 'local multiplayer', 'cross platform', 'cross progression',
            'new game plus', 'end game', 'post game', 'true ending',
            'speed run', 'glitchless run', 'any percent', 'hundred percent',
            'easter egg', 'secret level', 'hidden boss', 'unlockable character',
            'difficulty spike', 'difficulty curve', 'tutorial zone', 'starting area'
        ]
    
    def _compile_high_value_patterns(self):
        """Compile patterns for high-value content detection."""
        
        # Sections indiquant du contenu de haute qualité
        self.valuable_sections = re.compile(
            r'(?:gameplay|mechanics|combat system|progression system|'
            r'multiplayer|game modes|development|reception|legacy|'
            r'cultural impact|speedrunning|competitive play|'
            r'modding|community|metagame|strategy|tactics)',
            re.IGNORECASE
        )
        
        # Patterns de contenu approfondi
        self.depth_indicators = [
            re.compile(r'(?:for example|such as|including|specifically)', re.I),
            re.compile(r'(?:compared to|in contrast to|unlike|whereas)', re.I),
            re.compile(r'(?:originally|initially|later|eventually|subsequently)', re.I),
            re.compile(r'(?:however|although|despite|nevertheless)', re.I),
            re.compile(r'\b\d+(?:\.\d+)?%\b'),  # Pourcentages
            re.compile(r'\b\d+(?:st|nd|rd|th)\b'),  # Ordinaux
            re.compile(r'(?:patch|update|version)\s+\d+\.?\d*', re.I)
        ]
        
        # Patterns de contenu pauvre à pénaliser
        self.low_value_patterns = [
            re.compile(r'this article needs|citation needed|stub', re.I),
            re.compile(r'may refer to:|disambiguation', re.I),
            re.compile(r'list of|see also|external links', re.I),
            re.compile(r'[^\w\s]{10,}'),  # Trop de caractères spéciaux
        ]
    
    def score_document(self, document: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate quality score with detailed breakdown.
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        content = document.get('content', {})
        text = self._get_text(content)
        
        if not text or len(text) < 100:
            return 0.0, {}
        
        # Calculer les scores individuels
        scores = {
            'gaming_density': self._score_gaming_density_enhanced(text),
            'content_depth': self._score_content_depth(text, document),
            'structure': self._score_structure_enhanced(text),
            'uniqueness': self._score_uniqueness_enhanced(text, document),
            'length': self._score_length_optimized(len(text.split())),
            'freshness': self._score_freshness_enhanced(document)
        }
        
        # Score pondéré
        total_score = sum(
            scores[key] * getattr(self.weights, key)
            for key in scores
        )
        
        # Appliquer les bonus/malus
        total_score = self._apply_modifiers(total_score, document, scores)

        if total_score < 0.5:  # Debug les mauvais scores
            logger.debug(
                f"Low score breakdown for {document.get('document_id', 'unknown')[:8]}:\n"
                f"  - gaming_density: {scores['gaming_density']:.3f}\n"
                f"  - structure: {scores['structure']:.3f}\n"
                f"  - uniqueness: {scores['uniqueness']:.3f}\n"
                f"  - length: {scores['length']:.3f}\n"
                f"  - freshness: {scores['freshness']:.3f}\n"
                f"  - TOTAL: {total_score:.3f}"
            )
        
        return round(min(total_score, 1.0), 3), scores
    
    def _score_gaming_density_enhanced(self, text: str) -> float:
        """Enhanced gaming density scoring with category weights."""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        if word_count < 50:
            return 0.0
        
        # Scores par catégorie avec poids différents
        category_scores = {
            'basic_terms': 0,
            'gameplay_mechanics': 0,
            'technical_terms': 0,
            'esports_terms': 0,
            'high_value_phrases': 0
        }
        
        # Termes basiques (poids 1x)
        basic_gaming_count = sum(1 for word in words if word in self.gaming_terms)
        category_scores['basic_terms'] = min(basic_gaming_count / word_count * 10, 1.0)
        
        # Mécaniques de gameplay (poids 3x)
        mechanics_count = sum(1 for term in self.gameplay_mechanics if term in text_lower)
        category_scores['gameplay_mechanics'] = min(mechanics_count / 20, 1.0)  # Cap at 20 terms
        
        # Termes techniques (poids 2x)
        technical_count = sum(1 for term in self.technical_terms if term in text_lower)
        category_scores['technical_terms'] = min(technical_count / 15, 1.0)
        
        # Termes esports (poids 2x)
        esports_count = sum(1 for term in self.esports_terms if term in text_lower)
        category_scores['esports_terms'] = min(esports_count / 15, 1.0)
        
        # Phrases complexes (poids 4x)
        phrase_count = sum(1 for phrase in self.high_value_phrases if phrase in text_lower)
        category_scores['high_value_phrases'] = min(phrase_count / 10, 1.0)
        
        # Score pondéré final
        weighted_score = (
            category_scores['basic_terms'] * 1 +
            category_scores['gameplay_mechanics'] * 3 +
            category_scores['technical_terms'] * 2 +
            category_scores['esports_terms'] * 2 +
            category_scores['high_value_phrases'] * 4
        ) / 12  # Somme des poids
        
        # Bonus pour diversité de vocabulaire
        unique_gaming_terms = set()
        for word in words:
            if word in self.gaming_terms:
                unique_gaming_terms.add(word)
        
        diversity_bonus = min(len(unique_gaming_terms) / 50, 0.2)  # Max 0.2 bonus
        
        return min(weighted_score + diversity_bonus, 1.0)
    
    def _score_content_depth(self, text: str, document: Dict) -> float:
        """Score the depth and quality of content."""
        score = 0.0
        
        # 1. Présence de sections gaming importantes
        sections_found = self.valuable_sections.findall(text)
        section_score = min(len(sections_found) / 5, 0.3)  # Max 0.3
        score += section_score
        
        # 2. Indicateurs de profondeur (exemples, comparaisons, etc.)
        depth_count = 0
        for pattern in self.depth_indicators:
            depth_count += len(pattern.findall(text))
        depth_score = min(depth_count / 20, 0.3)  # Max 0.3
        score += depth_score
        
        # 3. Longueur moyenne des paragraphes (éviter les listes simples)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if 50 <= avg_para_length <= 200:  # Longueur idéale
                score += 0.2
            elif 30 <= avg_para_length < 50 or 200 < avg_para_length <= 300:
                score += 0.1
        
        # 4. Ratio explications/énumérations
        explanation_words = ['because', 'therefore', 'however', 'although', 'while', 
                           'when', 'if', 'unless', 'since', 'so that']
        explanation_count = sum(1 for word in explanation_words if word in text.lower())
        explanation_ratio = explanation_count / (len(text.split()) / 100)  # Per 100 words
        score += min(explanation_ratio / 5, 0.2)  # Max 0.2
        
        return min(score, 1.0)
    
    def _score_structure_enhanced(self, text: str) -> float:
        """Enhanced structure scoring focusing on gaming content organization."""
        score = 0.0
        
        # Patterns de structure
        patterns = {
            'sections': (r'^#+\s+.+$|^.+\n[=-]+$', 0.15),
            'gaming_sections': (r'(?:Gameplay|Story|Development|Reception|Modes)', 0.20),
            'lists': (r'^\s*[\*\-\d]+\.?\s+.+$', 0.10),
            'code_or_stats': (r'```[\s\S]*?```|`[^`\n]+`|\b\d+(?:\.\d+)?%', 0.10),
            'comparisons': (r'(?:compared to|versus|vs\.?|better than|worse than)', 0.15),
            'temporal': (r'(?:before|after|during|while|when|then)', 0.10),
            'examples': (r'(?:for example|e\.g\.|such as|including)', 0.15),
            'emphasis': (r'\*{1,2}[^*\n]+\*{1,2}|_{1,2}[^_\n]+_{1,2}', 0.05)
        }
        
        for pattern_name, (pattern, weight) in patterns.items():
            matches = len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
            if matches > 0:
                # Scoring non-linéaire pour éviter le spam
                if pattern_name in ['sections', 'gaming_sections']:
                    score += weight * min(matches / 8, 1.0)
                elif pattern_name == 'lists':
                    score += weight * min(matches / 20, 1.0)
                else:
                    score += weight * min(matches / 10, 1.0)
        
        return min(score, 1.0)
    
    def _score_uniqueness_enhanced(self, text: str, document: Dict) -> float:
        """Enhanced uniqueness scoring with better Wikipedia detection."""
        score = 1.0
        
        # Patterns Wikipedia spécifiques à pénaliser
        wikipedia_boilerplate = [
            r'\[edit\]', r'\[citation needed\]', r'needs additional citations',
            r'stub.*article', r'disambiguation page', r'may refer to:',
            r'needs to be updated', r'cleanup from', r'unreferenced',
            r'WikiProject', r'talk page', r'See also:', r'External links:',
            r'References:', r'Retrieved from', r'Categories:',
            r'needs attention from an expert', r'orphaned article'
        ]
        
        boilerplate_count = 0
        for pattern in wikipedia_boilerplate:
            if re.search(pattern, text, re.IGNORECASE):
                boilerplate_count += 1
                score -= 0.1
        
        # Pénaliser les listes simples sans contexte
        lines = text.split('\n')
        list_lines = sum(1 for line in lines if re.match(r'^\s*[\*\-\d]+\.?\s+', line))
        if lines:
            list_ratio = list_lines / len(lines)
            if list_ratio > 0.5:  # Plus de 50% de listes
                score -= 0.3
        
        # Bonus pour contenu original (citations de développeurs, interviews, etc.)
        original_content_patterns = [
            r'(?:said|stated|announced|revealed|explained)(?:\s+that)?',
            r'(?:according to|in an interview|during development)',
            r'(?:quote|"[^"]{20,}")',  # Citations longues
            r'(?:behind the scenes|making of|developer commentary)'
        ]
        
        original_count = 0
        for pattern in original_content_patterns:
            original_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        score += min(original_count / 10, 0.3)  # Bonus max 0.3
        
        return max(0.0, min(score, 1.0))
    
    def _score_length_optimized(self, word_count: int) -> float:
        """Optimized length scoring for gaming content."""
        # Courbe optimisée pour le contenu gaming
        if word_count < 200:
            return word_count / 200 * 0.3  # Max 0.3 pour contenu court
        elif word_count < 500:
            return 0.3 + (word_count - 200) / 300 * 0.3  # 0.3 à 0.6
        elif word_count < 2000:
            return 0.6 + (word_count - 500) / 1500 * 0.3  # 0.6 à 0.9
        elif word_count < 5000:
            return 0.9 + (word_count - 2000) / 3000 * 0.1  # 0.9 à 1.0
        elif word_count < 10000:
            return 1.0  # Parfait
        else:
            # Pénalité légère pour contenu très long
            return max(0.8, 1.0 - (word_count - 10000) / 50000)
    
    def _score_freshness_enhanced(self, document: Dict) -> float:
        """Enhanced freshness scoring for gaming content."""
        score = 0.5  # Base neutre
        
        metadata = document.get('metadata', {})
        
        # Extraire info du jeu
        game_info = self._get_game_info(metadata)
        release_year = self._extract_year(game_info.get('release_date', ''))
        
        # Scoring basé sur l'année avec courbe ajustée
        if release_year:
            current_year = 2024
            age = current_year - release_year
            
            if age <= 1:
                score = 1.0
            elif age <= 3:
                score = 0.9
            elif age <= 5:
                score = 0.8
            elif age <= 10:
                score = 0.6 + (10 - age) / 10 * 0.2
            elif age <= 20:
                score = 0.4 + (20 - age) / 20 * 0.2
            else:
                score = 0.3  # Minimum pour les classiques
        
        # Ajustements selon le type de contenu
        content_type = self._get_content_type(metadata)
        
        if content_type in ['guide', 'tutorial', 'strategy']:
            # Les guides vieillissent plus vite
            score *= 0.8
        elif content_type in ['encyclopedia', 'history', 'retrospective']:
            # Le contenu historique vieillit mieux
            score = min(score * 1.2, 1.0)
        
        # Bonus pour jeux avec contenu actif
        tags = self._get_tags(metadata)
        if any(tag in tags for tag in ['live-service', 'multiplayer', 'esports', 'mmo']):
            score = min(score * 1.3, 1.0)
        
        # Bonus pour les franchises actives
        if 'franchise' in str(document.get('content', {}).get('title', '')).lower():
            score = min(score * 1.1, 1.0)
        
        return score
    
    def _apply_modifiers(self, base_score: float, document: Dict, 
                        component_scores: Dict[str, float]) -> float:
        """Apply bonus/penalty modifiers to the base score."""
        score = base_score
        
        # 1. Bonus synergie (si plusieurs composants sont excellents)
        excellent_components = sum(1 for s in component_scores.values() if s >= 0.8)
        if excellent_components >= 3:
            score *= 1.1
        elif excellent_components >= 4:
            score *= 1.15
        
        # 2. Pénalité pour contenu de mauvaise qualité
        poor_components = sum(1 for s in component_scores.values() if s < 0.4)
        if poor_components >= 3:
            score *= 0.85
        
        # 3. Bonus pour combinaisons spécifiques
        if (component_scores.get('gaming_density', 0) >= 0.7 and 
            component_scores.get('content_depth', 0) >= 0.7):
            score *= 1.05  # Excellent contenu gaming approfondi
        
        # 4. Modificateurs basés sur les métadonnées
        metadata = document.get('metadata', {})
        
        # Bonus pour articles Wikipedia de qualité
        if hasattr(metadata, 'source'):
            source = metadata.source
        else:
            source = metadata.get('source', {})
            
        if isinstance(source, dict) and source.get('platform') == 'wikipedia':
            title = document.get('content', {}).get('title', '')
            # Featured articles
            if any(indicator in title for indicator in ['Featured article', 'Good article']):
                score *= 1.1
        
        # 5. Pénalité pour patterns de basse qualité détectés
        text = self._get_text(document.get('content', {}))
        for pattern in self.low_value_patterns:
            if pattern.search(text):
                score *= 0.95
        
        return min(score, 1.0)
    
    def batch_score_with_insights(self, documents: List[Dict]) -> List[Tuple[float, Dict]]:
        """
        Score multiple documents and provide insights.
        
        Returns:
            List of (score, insights) tuples
        """
        results = []
        
        for doc in documents:
            score, components = self.score_document(doc)
            
            # Générer des insights
            insights = {
                'score': score,
                'components': components,
                'strengths': [],
                'weaknesses': [],
                'suggestions': []
            }
            
            # Identifier forces et faiblesses
            for component, value in components.items():
                if value >= 0.8:
                    insights['strengths'].append(f"Excellent {component.replace('_', ' ')}")
                elif value < 0.5:
                    insights['weaknesses'].append(f"Poor {component.replace('_', ' ')}")
            
            # Suggestions d'amélioration
            if components.get('gaming_density', 0) < 0.6:
                insights['suggestions'].append("Add more gaming-specific terminology and mechanics discussion")
            if components.get('content_depth', 0) < 0.6:
                insights['suggestions'].append("Include more examples, comparisons, and detailed explanations")
            if components.get('structure', 0) < 0.6:
                insights['suggestions'].append("Improve organization with clear sections and better formatting")
            
            results.append((score, insights))
        
        return results
    
    def get_vocabulary_coverage(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze gaming vocabulary coverage across documents."""
        all_gaming_terms_found = set()
        term_frequency = Counter()
        category_coverage = {
            'basic': set(),
            'mechanics': set(),
            'technical': set(),
            'esports': set()
        }
        
        for doc in documents:
            text = self._get_text(doc.get('content', {})).lower()
            words = text.split()
            
            # Basic terms
            for word in words:
                if word in self.gaming_terms:
                    all_gaming_terms_found.add(word)
                    term_frequency[word] += 1
                    category_coverage['basic'].add(word)
            
            # Specialized terms
            for term in self.gameplay_mechanics:
                if term in text:
                    category_coverage['mechanics'].add(term)
                    term_frequency[term] += 1
            
            for term in self.technical_terms:
                if term in text:
                    category_coverage['technical'].add(term)
                    term_frequency[term] += 1
                    
            for term in self.esports_terms:
                if term in text:
                    category_coverage['esports'].add(term)
                    term_frequency[term] += 1
        
        # Calculer les statistiques
        total_vocabulary = (
            len(self.gaming_terms) + 
            len(self.gameplay_mechanics) + 
            len(self.technical_terms) + 
            len(self.esports_terms)
        )
        
        coverage_stats = {
            'total_unique_terms': len(all_gaming_terms_found),
            'vocabulary_coverage': len(all_gaming_terms_found) / total_vocabulary,
            'category_coverage': {
                cat: len(terms) / len(getattr(self, f"{cat}_terms" if cat != 'mechanics' else "gameplay_mechanics", set()))
                for cat, terms in category_coverage.items()
                if cat != 'basic'  # Basic uses gaming_terms which is loaded differently
            },
            'most_common_terms': term_frequency.most_common(50),
            'rare_terms': [term for term, count in term_frequency.items() if count == 1][:50]
        }
        
        return coverage_stats
    
    def _get_text(self, content: Any) -> str:
        """Extract text from content object or dict."""
        if hasattr(content, 'text_clean'):
            return content.text_clean or content.text or ''
        elif isinstance(content, dict):
            return content.get('text_clean') or content.get('text', '')
        return ''
    
    def _get_game_info(self, metadata: Any) -> Dict:
        """Extract game info from metadata."""
        if hasattr(metadata, 'game'):
            game = metadata.game
            if hasattr(game, 'to_dict'):
                return game.to_dict()
            return {'release_date': getattr(game, 'release_date', '')}
        elif isinstance(metadata, dict):
            return metadata.get('game', {})
        return {}
    
    def _get_content_type(self, metadata: Any) -> str:
        """Extract content type from metadata."""
        if hasattr(metadata, 'classification'):
            classification = metadata.classification
            if hasattr(classification, 'content_type'):
                return str(classification.content_type.value if hasattr(classification.content_type, 'value') else classification.content_type)
        elif isinstance(metadata, dict):
            return metadata.get('classification', {}).get('content_type', '')
        return ''
    
    def _get_tags(self, metadata: Any) -> List[str]:
        """Extract tags from metadata."""
        if hasattr(metadata, 'classification'):
            return getattr(metadata.classification, 'tags', [])
        elif isinstance(metadata, dict):
            return metadata.get('classification', {}).get('tags', [])
        return []
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        
        year_match = re.search(r'(19|20)\d{2}', str(date_str))
        if year_match:
            return int(year_match.group())
        return None