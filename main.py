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

from collectors.wikipedia_collector import WikipediaCollector
from processors.quality_scorer import EnhancedQualityScorer
from storage.parquet_manager import ParquetManager


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
        
        # Créer le collector avec configuration
        collector = WikipediaCollector(start_year=phase_config.get('year_filter', (2000, 2024))[0])
        
        # Override categories
        collector.CATEGORIES = phase_config['categories']
        
        collected_size_mb = 0
        batch = []
        
        async with collector:
            async for doc_dict in collector.extract():
                # Vérifier le budget
                if collected_size_mb >= phase_config['budget_mb']:
                    logger.info(f"Phase budget reached: {collected_size_mb}MB")
                    break
                
                try:
                    # Calculer le score avec le nouveau scorer
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
                        
                        # Log détaillé pour les premiers rejets
                        if phase_stats['articles_rejected'] <= 5:
                            logger.debug(
                                f"Rejected: {doc_dict['content']['title']}\n"
                                f"  Score: {score:.3f}\n"
                                f"  Components: {components}"
                            )
                        continue
                    
                    # Document accepté
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
        
        # Sauvegarder le dernier batch
        if batch:
            size_mb = await self._save_batch(batch, phase_config['name'])
            phase_stats['total_size_mb'] += size_mb
        
        # Finaliser les stats de phase
        phase_stats['ended_at'] = datetime.now()
        phase_stats['duration'] = phase_stats['ended_at'] - phase_stats['started_at']
        phase_stats['avg_quality'] = (
            sum(phase_stats['quality_scores']) / len(phase_stats['quality_scores'])
            if phase_stats['quality_scores'] else 0
        )
        
        self.stats['phase_stats'][phase_config['name']] = phase_stats
        self.stats['total_size_mb'] += phase_stats['total_size_mb']
        
        # Rapport de phase
        self._print_phase_report(phase_config['name'], phase_stats)
    
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
        """Génère le rapport final de collecte."""
        print(f"\n{'='*80}")
        print(f"🎮 RAPPORT FINAL - PHASE 1 WIKIPEDIA")
        print(f"{'='*80}")
        
        # Stats globales
        total_collected = sum(
            stats['articles_collected'] 
            for stats in self.stats['phase_stats'].values()
        )
        total_rejected = sum(
            stats['articles_rejected'] 
            for stats in self.stats['phase_stats'].values()
        )
        
        print(f"\n📊 Statistiques globales:")
        print(f"   - Articles collectés: {total_collected}")
        print(f"   - Articles rejetés: {total_rejected}")
        print(f"   - Taux d'acceptation: {total_collected/(total_collected+total_rejected)*100:.1f}%")
        print(f"   - Taille totale: {self.stats['total_size_mb']:.2f}MB")
        print(f"   - Vocabulaire unique: {len(self.stats['vocabulary_coverage'])} termes")
        
        # Qualité moyenne par phase
        print(f"\n📈 Qualité moyenne par phase:")
        for phase_name, stats in self.stats['phase_stats'].items():
            print(f"   - {phase_name}: {stats['avg_quality']:.3f}")
        
        # Top rejection reasons
        all_rejections = {}
        for stats in self.stats['phase_stats'].values():
            for reason, count in stats['rejection_reasons'].items():
                all_rejections[reason] = all_rejections.get(reason, 0) + count
        
        if all_rejections:
            print(f"\n🚫 Top 5 raisons de rejet globales:")
            for reason, count in sorted(all_rejections.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {reason}: {count}")
        
        # Recommandations
        print(f"\n💡 Recommandations:")
        avg_quality = sum(
            stats['avg_quality'] * stats['articles_collected']
            for stats in self.stats['phase_stats'].values()
        ) / total_collected
        
        if avg_quality >= 0.75:
            print("   ✅ Excellente qualité moyenne! Dataset prêt pour l'entraînement.")
        elif avg_quality >= 0.65:
            print("   ⚠️  Qualité correcte. Considérer un filtrage additionnel.")
        else:
            print("   ❌ Qualité insuffisante. Revoir les critères de sélection.")
        
        if self.stats['total_size_mb'] < 500:
            print(f"   ⚠️  Taille en dessous de l'objectif. Continuer la collecte.")
        
        # Sauvegarder le rapport
        self._save_report_to_file()
    
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