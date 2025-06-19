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


# Dans quality_scorer.py

class EnhancedQualityScorer:
    def __init__(self):
        # Ajuster les poids - réduire uniqueness, augmenter relevance
        self.weights = {
            'relevance': 0.35,      # ↑ de 0.30
            'informativeness': 0.25, # → inchangé
            'coherence': 0.20,      # → inchangé  
            'uniqueness': 0.10,     # ↓ de 0.20
            'length': 0.10          # → inchangé
        }
        
        # Nouveaux seuils plus réalistes
        self.thresholds = {
            'min_length': 500,           # ↓ de 500 caractères
            'optimal_length': 5000,      # ↓ de 10000
            'min_gaming_density': 0.02,  # ↓ de 0.05 (2% minimum)
            'optimal_gaming_density': 0.08, # ↓ de 0.30 (8% optimal)
            'min_sections': 2,           # ↓ de 3
            'optimal_sections': 5        # ↓ de 8
        }
        
        # Termes gaming plus variés et réalistes
        self.gaming_terms = {
            # Termes génériques (poids faible)
            'game': 1.0, 'play': 0.8, 'player': 1.0, 'video': 0.5,
            
            # Termes spécifiques (poids fort)
            'gameplay': 2.0, 'multiplayer': 2.0, 'single-player': 2.0,
            'co-op': 2.0, 'pvp': 2.0, 'pve': 2.0, 'fps': 2.0,
            
            # Mécaniques
            'level': 1.5, 'quest': 1.5, 'mission': 1.5, 'boss': 1.5,
            'character': 1.2, 'skill': 1.2, 'ability': 1.2, 'upgrade': 1.5,
            
            # Genres
            'rpg': 2.0, 'mmorpg': 2.0, 'rts': 2.0, 'moba': 2.0,
            'platformer': 2.0, 'roguelike': 2.0, 'sandbox': 2.0,
            
            # Plateformes
            'playstation': 1.5, 'xbox': 1.5, 'nintendo': 1.5, 'steam': 1.5,
            'pc': 1.0, 'console': 1.2, 'mobile': 1.0,
            
            # Développement
            'developer': 1.5, 'publisher': 1.5, 'studio': 1.2,
            'release': 1.0, 'launch': 1.0, 'port': 1.2,
            
            # Compétitif
            'esports': 2.0, 'tournament': 1.5, 'competitive': 1.5,
            'leaderboard': 1.5, 'ranking': 1.2, 'matchmaking': 2.0
        }

    def _calculate_relevance(self, text: str, metadata: Dict) -> Tuple[float, Dict]:
        """Calcul plus nuancé de la pertinence gaming."""
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0.0, {'gaming_density': 0.0}
        
        # Calculer le score pondéré des termes gaming
        gaming_score = 0.0
        gaming_terms_found = []
        
        for word in words:
            for term, weight in self.gaming_terms.items():
                if term in word:  # Permet "gameplay" de matcher dans "gameplaywise"
                    gaming_score += weight
                    if term not in gaming_terms_found:
                        gaming_terms_found.append(term)
        
        # Normaliser par le nombre de mots
        gaming_density = gaming_score / (total_words * 2.0)  # Divisé par 2 pour normaliser
        
        # Bonus pour les catégories Wikipedia
        category_bonus = 0.0
        categories = metadata.get('categories', [])
        gaming_categories = [
            'video_game', 'esports', 'gaming', 'playstation', 'xbox', 
            'nintendo', 'steam', 'multiplayer', 'single-player'
        ]
        
        for cat in categories:
            cat_lower = cat.lower()
            if any(gc in cat_lower for gc in gaming_categories):
                category_bonus = 0.2
                break
        
        # Score avec seuils ajustés
        if gaming_density < self.thresholds['min_gaming_density']:
            base_score = gaming_density / self.thresholds['min_gaming_density'] * 0.5
        elif gaming_density < self.thresholds['optimal_gaming_density']:
            # Progression linéaire de 0.5 à 1.0
            progress = (gaming_density - self.thresholds['min_gaming_density']) / \
                      (self.thresholds['optimal_gaming_density'] - self.thresholds['min_gaming_density'])
            base_score = 0.5 + (progress * 0.5)
        else:
            # Plateau après l'optimal avec légère décroissance
            base_score = 1.0 - ((gaming_density - self.thresholds['optimal_gaming_density']) * 0.5)
            base_score = max(0.8, base_score)  # Minimum 0.8 si très dense
        
        final_score = min(1.0, base_score + category_bonus)
        
        return final_score, {
            'gaming_density': gaming_density,
            'gaming_terms_found': gaming_terms_found[:10],  # Top 10 pour debug
            'category_bonus': category_bonus
        }

    def _calculate_length_score(self, text: str) -> Tuple[float, Dict]:
        """Score de longueur plus tolérant."""
        length = len(text)
        
        if length < self.thresholds['min_length']:
            # Pénalité progressive, pas brutale
            score = (length / self.thresholds['min_length']) ** 0.5  # Racine carrée pour être moins sévère
        elif length < self.thresholds['optimal_length']:
            # Progression douce vers l'optimal
            progress = (length - self.thresholds['min_length']) / \
                      (self.thresholds['optimal_length'] - self.thresholds['min_length'])
            score = 0.7 + (progress * 0.3)
        else:
            # Les articles longs sont OK
            score = 1.0
        
        return score, {'text_length': length}

    def _calculate_informativeness(self, document: Dict) -> Tuple[float, Dict]:
        """Informativeness ajustée pour Wikipedia."""
        content = document.get('content', {})
        metadata = document.get('metadata', {})
        
        # Nombre de sections
        sections = content.get('sections', {})
        section_count = len(sections)
        
        if section_count < self.thresholds['min_sections']:
            section_score = section_count / self.thresholds['min_sections'] * 0.7
        else:
            section_score = min(1.0, 0.7 + (section_count / self.thresholds['optimal_sections'] * 0.3))
        
        # Présence d'infobox
        has_infobox = bool(metadata.get('game_info', {}))
        infobox_score = 1.0 if has_infobox else 0.7
        
        # Sections importantes pour le gaming
        important_sections = ['gameplay', 'development', 'reception', 'plot', 'story']
        important_count = sum(1 for s in important_sections if s in sections)
        important_score = min(1.0, 0.5 + (important_count * 0.1))
        
        # Score combiné
        final_score = (section_score * 0.4 + infobox_score * 0.3 + important_score * 0.3)
        
        return final_score, {
            'section_count': section_count,
            'has_infobox': has_infobox,
            'important_sections': important_count
        }
    
    # Dans quality_scorer.py - Ajouter cette méthode à EnhancedQualityScorer

    def score_document(self, document: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate quality score with detailed breakdown.
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        content = document.get('content', {})
        metadata = document.get('metadata', {})
        text = self._get_text(content)
        
        # Debug logging
        doc_id = document.get('document_id', 'unknown')[:8]
        logger.debug(f"Scoring document {doc_id}, text length: {len(text)} chars")
        
        # Si texte trop court, retourner un score faible mais pas 0
        if not text or len(text) < 100:
            logger.warning(f"Document {doc_id} has very short text: {len(text)} chars")
            return 0.1, {'error': 'text_too_short'}
        
        # Calculer les scores individuels avec les nouvelles méthodes
        relevance_score, relevance_details = self._calculate_relevance(text, metadata)
        length_score, length_details = self._calculate_length_score(text)
        informativeness_score, info_details = self._calculate_informativeness(document)
        
        # Pour coherence et uniqueness, utiliser des valeurs par défaut pour l'instant
        coherence_score = 0.8  # Wikipedia est généralement cohérent
        uniqueness_score = 0.7  # Pas de duplicatas sur Wikipedia
        
        # Dictionnaire des scores
        scores = {
            'relevance': relevance_score,
            'informativeness': informativeness_score,
            'coherence': coherence_score,
            'uniqueness': uniqueness_score,
            'length': length_score
        }
        
        # Score pondéré selon les nouveaux poids
        total_score = sum(
            scores[key] * self.weights[key]
            for key in scores
        )
        
        # Appliquer des bonus/malus contextuels
        total_score = self._apply_modifiers(total_score, document, scores)
        
        # Debug pour les scores faibles
        if total_score < 0.5:
            logger.debug(
                f"Low score breakdown for {doc_id}:\n"
                f"  - relevance: {relevance_score:.3f} (density: {relevance_details.get('gaming_density', 0):.3f})\n"
                f"  - informativeness: {informativeness_score:.3f}\n"
                f"  - length: {length_score:.3f} ({length_details.get('text_length', 0)} chars)\n"
                f"  - TOTAL: {total_score:.3f}"
            )
        
        # Retourner le score et les détails
        component_scores = {
            **scores,
            **relevance_details,
            **length_details,
            **info_details
        }
        
        return round(min(total_score, 1.0), 3), component_scores

    def _apply_modifiers(self, base_score: float, document: Dict, 
                        component_scores: Dict[str, float]) -> float:
        """Apply bonus/penalty modifiers to the base score."""
        score = base_score
        
        # Bonus pour articles de qualité Wikipedia
        metadata = document.get('metadata', {})
        categories = metadata.get('categories', [])
        
        # Featured/Good articles
        if any('featured' in cat.lower() for cat in categories):
            score *= 1.15
        elif any('good' in cat.lower() for cat in categories):
            score *= 1.10
        
        # Pénalité pour stubs
        if any('stub' in cat.lower() for cat in categories):
            score *= 0.8
        
        # Bonus synergie si plusieurs composants sont bons
        good_components = sum(1 for s in component_scores.values() if isinstance(s, float) and s >= 0.7)
        if good_components >= 3:
            score *= 1.05
        
        return min(score, 1.0)

    def _get_text(self, content: Any) -> str:
        """Extract text from content object or dict."""
        text = ''
        
        # Try different extraction methods
        if hasattr(content, 'text_clean'):
            text = content.text_clean or content.text or ''
        elif isinstance(content, dict):
            # Try multiple possible keys
            text = (content.get('text_clean') or 
                content.get('text') or 
                content.get('summary') or '')
            
            # Si on a des sections, les concatener aussi
            if 'sections' in content and isinstance(content['sections'], dict):
                sections_text = ' '.join(content['sections'].values())
                if sections_text:
                    text = text + '\n\n' + sections_text
        
        return text