#!/usr/bin/env python3
"""
Stratégie de collecte optimisée pour la Phase 1 Wikipedia.
"""

import asyncio
import click
from pathlib import Path
from datetime import datetime
from loguru import logger
import json
from typing import Dict, List
from collectors.wikipedia_collector import WikipediaCollector, WikipediaConfig
from processors.quality_scorer import EnhancedQualityScorer
from storage.parquet_manager import ParquetManager

class YearFilter:
    def __init__(self, min_year: int, max_year: int):
        self.min_year = min_year
        self.max_year = max_year
    
    def apply(self, doc: Dict) -> bool:
        game_info = doc.get('metadata', {}).get('game_info', {})
        release_year = game_info.get('release_year')
        if release_year:
            return self.min_year <= release_year <= self.max_year
        return True  # Si pas d'année, on laisse passer

class ContentRequirementFilter:
    def __init__(self, required_terms: List[str]):
        self.required_terms = required_terms
    
    def apply(self, doc: Dict) -> bool:
        text = doc.get('content', {}).get('text_clean', '').lower()
        return all(term in text for term in self.required_terms)
    
class OptimizedCollectionStrategy:
    """Stratégie de collecte intelligente avec priorisation."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.storage = ParquetManager(output_dir)
        self.scorer = EnhancedQualityScorer()
        
        # Statistiques de collecte
        self.stats = {
            'phase_stats': {},
            'quality_distribution': {},
            'vocabulary_coverage': set(),
            'total_size_mb': 0
        }
    
    async def execute_phase_1(self, target_size_mb: int = 750):
        """
        Exécute la collecte Phase 1 avec stratégie optimisée.
        
        Args:
            target_size_mb: Taille cible en MB
        """
        logger.info(f"Starting Phase 1 collection - Target: {target_size_mb}MB")
        
        # Phases de collecte avec budgets
        collection_phases = [
            {
                'name': 'Phase 1A - High Priority',
                'budget_mb': 300,
                'min_quality': 0.7,
                'categories': [
                    'Category:Featured_video_game_articles',
                    'Category:Good_video_game_articles',
                    'Category:2024_video_games',
                    'Category:2023_video_games',
                    'Category:The_Game_Award_for_Game_of_the_Year_winners'
                ],
                'year_filter': (2020, 2024)
            },
            {
                'name': 'Phase 1B - Gaming Mechanics',
                'budget_mb': 150,
                'min_quality': 0.65,
                'categories': [
                    'Category:Video_game_gameplay',
                    'Category:Video_game_mechanics', 
                    'Category:Video_game_terminology',
                    'Category:Video_game_genres'
                ],
                'content_requirements': ['gameplay', 'mechanics']
            },
            {
                'name': 'Phase 1C - Franchises & Series',
                'budget_mb': 150,
                'min_quality': 0.7,
                'categories': [
                    'Category:Video_game_franchises',
                    'Category:Video_game_sequels',
                    'Category:Long-running_video_game_franchises'
                ],
                'prefer_series': True
            },
            {
                'name': 'Phase 1D - Competitive & Esports',
                'budget_mb': 100,
                'min_quality': 0.65,
                'categories': [
                    'Category:Esports_games',
                    'Category:Fighting_games',
                    'Category:Multiplayer_online_battle_arena_games',
                    'Category:First-person_shooters'
                ],
                'tags_required': ['multiplayer', 'competitive']
            },
            {
                'name': 'Phase 1E - Indies & Innovation',
                'budget_mb': 50,
                'min_quality': 0.7,
                'categories': [
                    'Category:Indie_video_games',
                    'Category:IGF_Grand_Prize_winners',
                    'Category:IndieCade_winners'
                ]
            }
        ]
        
        # Exécuter chaque phase
        for phase in collection_phases:
            await self._execute_collection_phase(phase)
            
            # Vérifier si on a atteint la taille cible
            if self.stats['total_size_mb'] >= target_size_mb:
                logger.info(f"Target size reached: {self.stats['total_size_mb']}MB")
                break
        
        # Rapport final
        self._generate_final_report()
    
    async def _execute_collection_phase(self, phase_config: dict):
        """Exécute une phase de collecte spécifique."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Executing: {phase_config['name']}")
        logger.info(f"Budget: {phase_config['budget_mb']}MB")
        logger.info(f"Min quality: {phase_config['min_quality']}")
        logger.info(f"{'='*60}")
        
        phase_stats = {
            'started_at': datetime.now(),
            'articles_collected': 0,
            'articles_rejected': 0,
            'total_size_mb': 0,
            'quality_scores': [],
            'rejection_reasons': {}
        }
        
        # 1. D'abord créer la configuration du collector
        config = WikipediaConfig(
            language='en',
            categories_per_batch=10,
            pages_per_category=100,
            min_article_length=1000,
            extract_infobox=True
        )
        
        # 2. Créer le collector
        collector = WikipediaCollector(config)
        
        # 3. Override les catégories pour cette phase
        if 'categories' in phase_config:
            collector.PRIORITY_CATEGORIES = phase_config['categories']
        
        collected_size_mb = 0
        batch = []
        
        # 4. Collecter les documents
        async with collector:
            async for doc_dict in collector.extract():
                # Vérifier le budget
                if collected_size_mb >= phase_config['budget_mb']:
                    logger.info(f"Phase budget reached: {collected_size_mb}MB")
                    break
                
                try:
                    # 5. MAINTENANT appliquer les filtres sur le document récupéré
                    
                    # Calculer le score de qualité
                    score, components = self.scorer.score_document(doc_dict)
                    phase_stats['quality_scores'].append(score)
                    
                    # Appliquer les filtres spécifiques à la phase
                    if not self._passes_phase_filters(doc_dict, phase_config):
                        phase_stats['articles_rejected'] += 1
                        phase_stats['rejection_reasons']['phase_filter'] = \
                            phase_stats['rejection_reasons'].get('phase_filter', 0) + 1
                        continue
                    
                    # Vérifier la qualité
                    if score < phase_config['min_quality']:
                        phase_stats['articles_rejected'] += 1
                        reason = f"quality_below_{phase_config['min_quality']}"
                        phase_stats['rejection_reasons'][reason] = \
                            phase_stats['rejection_reasons'].get(reason, 0) + 1
                        continue
                    
                    # Document accepté !
                    doc_dict['quality'] = {
                        'overall': score,
                        **components
                    }
                    
                    batch.append(doc_dict)
                    phase_stats['articles_collected'] += 1
                    
                    # Log les bons articles
                    if score >= 0.8:
                        logger.info(
                            f"✓ High quality: {doc_dict['content']['title']} "
                            f"(score: {score:.3f})"
                        )
                    
                    # Sauvegarder par batch
                    if len(batch) >= 100:
                        size_mb = await self._save_batch(batch, phase_config['name'])
                        collected_size_mb += size_mb
                        phase_stats['total_size_mb'] += size_mb
                        batch = []
                        
                        # Mise à jour du vocabulaire
                        self._update_vocabulary_coverage(batch)
                        
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
                    phase_stats['articles_rejected'] += 1
                    phase_stats['rejection_reasons']['error'] = \
                        phase_stats['rejection_reasons'].get('error', 0) + 1
                    
    def _passes_phase_filters(self, doc_dict: dict, phase_config: dict) -> bool:
        """Vérifie si un document passe les filtres spécifiques à la phase."""
        content = doc_dict.get('content', {})
        metadata = doc_dict.get('metadata', {})
        
        # Filtre par année
        if 'year_filter' in phase_config:
            year_min, year_max = phase_config['year_filter']
            game_info = metadata.get('game', {})
            release_date = game_info.get('release_date', '')
            
            if release_date:
                import re
                year_match = re.search(r'(\d{4})', release_date)
                if year_match:
                    year = int(year_match.group(1))
                    if not (year_min <= year <= year_max):
                        return False
        
        # Exigences de contenu
        if 'content_requirements' in phase_config:
            text = content.get('text_clean', '').lower()
            if not all(req in text for req in phase_config['content_requirements']):
                return False
        
        # Préférence pour les séries
        if phase_config.get('prefer_series'):
            enhanced_meta = metadata.get('enhanced_metadata', {})
            if not enhanced_meta.get('is_franchise'):
                # Accepter quand même mais avec une probabilité réduite
                import random
                if random.random() > 0.3:  # 30% de chance d'accepter
                    return False
        
        # Tags requis
        if 'tags_required' in phase_config:
            classification = metadata.get('classification', {})
            tags = classification.get('tags', [])
            enhanced_tags = metadata.get('enhanced_metadata', {}).get('enhanced_tags', [])
            all_tags = tags + enhanced_tags
            
            if not any(tag in all_tags for tag in phase_config['tags_required']):
                return False
        
        return True
    
    async def _save_batch(self, batch: list, phase_name: str) -> float:
        """Sauvegarde un batch de documents."""
        import pandas as pd
        
        df = pd.DataFrame(batch)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase1_{phase_name.replace(' ', '_')}_{timestamp}.parquet"
        
        file_size = self.storage.save(df, filename)
        size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Saved batch: {len(batch)} docs, {size_mb:.2f}MB")
        return size_mb
    
    def _update_vocabulary_coverage(self, documents: list):
        """Met à jour la couverture du vocabulaire gaming."""
        for doc in documents:
            text = doc.get('content', {}).get('text_clean', '').lower()
            words = text.split()
            
            # Ajouter au vocabulaire global
            gaming_terms = [w for w in words if len(w) > 3 and w.isalpha()]
            self.stats['vocabulary_coverage'].update(gaming_terms[:100])  # Limiter pour perf
    
    def _print_phase_report(self, phase_name: str, phase_stats: dict):
        """Affiche le rapport d'une phase."""
        print(f"\n{'='*60}")
        print(f"📊 {phase_name} - Rapport")
        print(f"{'='*60}")
        print(f"⏱️  Durée: {phase_stats['duration']}")
        print(f"✅ Articles collectés: {phase_stats['articles_collected']}")
        print(f"❌ Articles rejetés: {phase_stats['articles_rejected']}")
        print(f"💾 Taille collectée: {phase_stats['total_size_mb']:.2f}MB")
        print(f"📈 Score qualité moyen: {phase_stats['avg_quality']:.3f}")
        
        if phase_stats['rejection_reasons']:
            print(f"\n🚫 Raisons de rejet:")
            for reason, count in sorted(phase_stats['rejection_reasons'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"   - {reason}: {count}")
        
        # Distribution de qualité
        if phase_stats['quality_scores']:
            scores = phase_stats['quality_scores']
            print(f"\n📊 Distribution qualité:")
            print(f"   - Excellent (0.8+): {sum(1 for s in scores if s >= 0.8)}")
            print(f"   - Bon (0.7-0.8): {sum(1 for s in scores if 0.7 <= s < 0.8)}")
            print(f"   - Moyen (0.6-0.7): {sum(1 for s in scores if 0.6 <= s < 0.7)}")
            print(f"   - Faible (<0.6): {sum(1 for s in scores if s < 0.6)}")
    
    
    def _generate_final_report(self):
        """Génère un rapport final de la collecte."""
        total_collected = sum(p['articles_collected'] for p in self.stats['phases'])
        total_rejected = sum(p['articles_rejected'] for p in self.stats['phases'])
        total_size = sum(p['total_size_mb'] for p in self.stats['phases'])
        
        # Éviter la division par zéro
        total_articles = total_collected + total_rejected
        acceptance_rate = (total_collected / total_articles * 100) if total_articles > 0 else 0.0
        
        print("\n" + "="*60)
        print("FINAL COLLECTION REPORT")
        print("="*60)
        print(f"Duration: {self.stats['duration']}")
        print(f"\nArticles Statistics:")
        print(f"   - Total collected: {total_collected:,}")
        print(f"   - Total rejected: {total_rejected:,}")
        print(f"   - Acceptance rate: {acceptance_rate:.1f}%")
        print(f"   - Total size: {total_size:.2f} MB")
        
        if total_collected > 0:
            print(f"   - Average size: {total_size/total_collected:.3f} MB/article")
        
        # Distribution de qualité seulement si on a des scores
        all_scores = []
        for phase in self.stats['phases']:
            all_scores.extend(phase.get('quality_scores', []))
        
        if all_scores:
            import numpy as np
            scores_array = np.array(all_scores)
            print(f"\nQuality Distribution:")
            print(f"   - Mean: {scores_array.mean():.3f}")
            print(f"   - Std: {scores_array.std():.3f}")
            print(f"   - Min: {scores_array.min():.3f}")
            print(f"   - Max: {scores_array.max():.3f}")
            
            # Percentiles
            percentiles = [25, 50, 75, 90, 95]
            print(f"\n   Percentiles:")
            for p in percentiles:
                value = np.percentile(scores_array, p)
                print(f"   - P{p}: {value:.3f}")
        else:
            print("\nNo quality scores available.")
        
        # Rejection reasons
        print(f"\nRejection Reasons:")
        all_reasons = {}
        for phase in self.stats['phases']:
            for reason, count in phase.get('rejection_reasons', {}).items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        
        if all_reasons:
            # Trier par fréquence
            sorted_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons:
                percentage = (count / total_rejected * 100) if total_rejected > 0 else 0
                print(f"   - {reason}: {count} ({percentage:.1f}%)")
        else:
            print("   - No rejections recorded")
        
        # Phase par phase
        print(f"\nPhase Breakdown:")
        for i, phase in enumerate(self.stats['phases']):
            print(f"\n   Phase {i+1}: {phase.get('name', 'Unknown')}")
            print(f"   - Duration: {phase.get('duration', 'N/A')}")
            print(f"   - Collected: {phase['articles_collected']:,}")
            print(f"   - Rejected: {phase['articles_rejected']:,}")
            print(f"   - Size: {phase['total_size_mb']:.2f} MB")
            
            # Moyenne de qualité pour cette phase
            if phase.get('quality_scores'):
                avg_quality = sum(phase['quality_scores']) / len(phase['quality_scores'])
                print(f"   - Avg quality: {avg_quality:.3f}")
        
        # Vocabulary coverage
        if hasattr(self, 'vocabulary_tracker') and self.vocabulary_tracker:
            unique_terms = len(self.vocabulary_tracker)
            print(f"\nVocabulary Coverage:")
            print(f"   - Unique gaming terms: {unique_terms:,}")
            
            # Top terms
            if unique_terms > 0:
                top_terms = sorted(
                    self.vocabulary_tracker.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]
                print(f"   - Top 20 terms:")
                for term, count in top_terms:
                    print(f"      - {term}: {count:,}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if acceptance_rate < 20:
            print("   ⚠️  Very low acceptance rate - consider relaxing quality criteria")
        if total_collected < 1000:
            print("   ⚠️  Low collection count - consider increasing budget or reducing filters")
        if total_size < 100:
            print("   ⚠️  Small dataset size - may need more collection phases")
    
        # Success indicators
        if acceptance_rate > 50 and total_collected > 5000:
            print("   ✅ Excellent collection efficiency!")
        if total_size > 500:
            print("   ✅ Good dataset size achieved")

    def _save_report_to_file(self):
        """Sauvegarde le rapport dans un fichier JSON."""
        report_path = self.output_dir / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")


@click.command()
@click.option('--output-dir', default='data/wikipedia_phase1', help='Output directory')
@click.option('--target-size-mb', default=750, help='Target size in MB')
@click.option('--dry-run', is_flag=True, help='Test run without saving')
def main(output_dir, target_size_mb, dry_run):
    """Execute optimized Wikipedia collection strategy."""
    if dry_run:
        logger.info("DRY RUN MODE - No data will be saved")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    strategy = OptimizedCollectionStrategy(output_path)
    
    # Run collection
    asyncio.run(strategy.execute_phase_1(target_size_mb))


if __name__ == "__main__":
    main()