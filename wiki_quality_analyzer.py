#!/usr/bin/env python3
"""
Wikipedia Quality Analyzer
Analyse exploratoire de la qualité des articles Wikipedia par thématique.
Génère des rapports détaillés en JSON avec tableaux récapitulatifs.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import pandas as pd
from loguru import logger
import numpy as np

from collectors.wikipedia_collector import WikipediaCollector, WikipediaConfig
from processors.quality_scorer import EnhancedQualityScorer


class WikipediaQualityAnalyzer:
    """Analyse la qualité des articles Wikipedia par thématique."""
    

    THEMES = {
        "Video_Game_Culture": {
            "name": "Gaming Culture Core",
            "categories": [
                "Category:Video_game_culture",
                "Category:Gaming_communities",
                "Category:Video_game_fandom"
            ],
            "sample_size": 150,
            "description": "Central hub for gaming subculture, social dynamics, and community formation patterns"
        },
        "Gaming_Memes_Viral": {
            "name": "Memes and Viral Phenomena",
            "categories": [
                "Category:Video_game_memes",
                "Category:Internet_memes_related_to_video_games",
                "Category:Viral_video_games"
            ],
            "sample_size": 100,
            "description": "Captures shared cultural language, viral moments, and zeitgeist phenomena like 'Press F'"
        },
        "Gaming_Controversies": {
            "name": "Gaming Controversies",
            "categories": [
                "Category:Video_game_controversies",
                "Category:Video_game_censorship",
                "Category:Video_game_culture_conflicts"
            ],
            "sample_size": 75,
            "description": "Cultural flashpoints, GamerGate, industry scandals, and community debates"
        },
        "Esports_Competitive": {
            "name": "Esports and Competitive Gaming",
            "categories": [
                "Category:Esports",
                "Category:Esports_tournaments",
                "Category:Esports_teams"
            ],
            "sample_size": 120,
            "description": "Professional gaming culture, tournaments, and spectator community dynamics"
        },
        "Streaming_Content_Creation": {
            "name": "Streaming and Content Creators",
            "categories": [
                "Category:Video_game_streaming",
                "Category:Twitch_(service)",
                "Category:Gaming_YouTubers"
            ],
            "sample_size": 80,
            "description": "Streaming culture, influencer impact, and parasocial relationships in gaming"
        },
        "Japanese_Gaming_Culture": {
            "name": "Japanese Gaming Ecosystem",
            "categories": [
                "Category:Video_games_developed_in_Japan",
                "Category:Japanese_role-playing_video_games",
                "Category:Video_game_companies_of_Japan"
            ],
            "sample_size": 100,
            "description": "JRPG culture, otaku integration, visual novels, and unique Japanese gaming discourse"
        },
        "Korean_Gaming_Esports": {
            "name": "Korean Gaming Dominance",
            "categories": [
                "Category:Video_games_developed_in_South_Korea",
                "Category:Esports_in_South_Korea",
                "Category:PC_bangs"
            ],
            "sample_size": 60,
            "description": "PC bang culture, StarCraft legacy, professional gaming infrastructure"
        },
        "Gaming_Conventions_Events": {
            "name": "Gaming Events and Gatherings",
            "categories": [
                "Category:Gaming_conventions",
                "Category:Video_game_trade_shows",
                "Category:Gaming_events"
            ],
            "sample_size": 50,
            "description": "Physical community spaces, E3, PAX, regional events, and announcement culture"
        },
        "Online_Gaming_Communities": {
            "name": "Online Multiplayer Culture",
            "categories": [
                "Category:Online_games",
                "Category:Massively_multiplayer_online_games",
                "Category:Battle_royale_games"
            ],
            "sample_size": 90,
            "description": "Online interaction dynamics, toxicity, cooperation, and emergent social behaviors"
        },
        "Gaming_Journalism_Media": {
            "name": "Gaming Media and Criticism",
            "categories": [
                "Category:Video_game_journalism",
                "Category:Video_game_websites",
                "Category:Video_game_critics"
            ],
            "sample_size": 60,
            "description": "Gaming press influence, review culture, and media-community relationships"
        },
        "Indie_Gaming_Culture": {
            "name": "Indie Game Movement",
            "categories": [
                "Category:Indie_video_games",
                "Category:Video_game_development",
                "Category:Crowdfunded_video_games"
            ],
            "sample_size": 70,
            "description": "Alternative gaming culture, creative expression, and anti-corporate sentiment"
        },
        "Mobile_Gaming_Revolution": {
            "name": "Mobile and Casual Gaming",
            "categories": [
                "Category:Mobile_games",
                "Category:Free-to-play_video_games",
                "Category:Gacha_games"
            ],
            "sample_size": 80,
            "description": "Casual gaming demographics, monetization debates, and accessibility culture"
        },
        "Gaming_Hardware_Culture": {
            "name": "Platform Wars and Hardware",
            "categories": [
                "Category:Video_game_consoles",
                "Category:Console_wars",
                "Category:Gaming_computers"
            ],
            "sample_size": 60,
            "description": "Console tribalism, PC master race culture, and hardware enthusiasm"
        },
        "Retro_Gaming_Nostalgia": {
            "name": "Retro and Preservation",
            "categories": [
                "Category:Retro_gaming",
                "Category:Video_game_remakes",
                "Category:Video_game_preservation"
            ],
            "sample_size": 50,
            "description": "Nostalgia culture, speedrunning community, and game preservation movements"
        },
        "Gaming_Subcultures": {
            "name": "Specific Gaming Tribes",
            "categories": [
                "Category:Speedrunning",
                "Category:Modding",
                "Category:Let's_Play"
            ],
            "sample_size": 60,
            "description": "Niche communities with unique languages, practices, and cultural norms"
        },
        "Chinese_Gaming_Market": {
            "name": "Chinese Gaming Phenomenon",
            "categories": [
                "Category:Video_games_developed_in_China",
                "Category:Chinese_online_games",
                "Category:Tencent_games"
            ],
            "sample_size": 50,
            "description": "Mobile-first culture, gacha economics, and government regulation impacts"
        },
        "Game_Genres_Culture": {
            "name": "Genre-Specific Communities",
            "categories": [
                "Category:Fighting_games",
                "Category:First-person_shooters",
                "Category:Role-playing_video_games"
            ],
            "sample_size": 100,
            "description": "Genre-specific slang, community norms, and competitive scenes"
        },
        "Gaming_Ethics_Philosophy": {
            "name": "Gaming Ethics and Impact",
            "categories": [
                "Category:Video_game_addiction",
                "Category:Violence_in_video_games",
                "Category:Ethics_of_video_games"
            ],
            "sample_size": 40,
            "description": "Debates on gaming's societal impact, addiction concerns, and moral panics"
        },
        "Virtual_Worlds_Metaverse": {
            "name": "Virtual Worlds and Social Spaces",
            "categories": [
                "Category:Virtual_worlds",
                "Category:Social_simulation_video_games",
                "Category:Sandbox_games"
            ],
            "sample_size": 60,
            "description": "Minecraft culture, Roblox communities, and emergent social environments"
        },
        "Gaming_Academia_Research": {
            "name": "Gaming Studies and Research",
            "categories": [
                "Category:Video_game_studies",
                "Category:Video_game_researchers",
                "Category:Ludology"
            ],
            "sample_size": 30,
            "description": "Academic gaming discourse, research terminology, and theoretical frameworks"
        }
    }

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = EnhancedQualityScorer()
        self.results = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_articles_analyzed": 0,
                "themes_analyzed": []
            },
            "articles_by_theme": {},
            "summary_by_theme": {},
            "global_statistics": {}
        }
        
    async def analyze_all_themes(self):
        """Analyse toutes les thématiques définies."""
        logger.info(f"Starting quality analysis for {len(self.THEMES)} themes")
        
        all_articles_data = []
        
        for theme_id, theme_config in self.THEMES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing theme: {theme_config['name']}")
            logger.info(f"Description: {theme_config.get('description', 'No description')}")
            logger.info(f"Target sample size: {theme_config.get('sample_size', 'Not specified')}")
            
            theme_results = await self._analyze_theme(theme_id, theme_config)
            
            # Stocker les résultats
            self.results["articles_by_theme"][theme_id] = theme_results["articles"]
            self.results["summary_by_theme"][theme_id] = theme_results["summary"]
            self.results["analysis_metadata"]["themes_analyzed"].append(theme_id)
            
            # Ajouter pour les stats globales
            all_articles_data.extend(theme_results["articles"])
            
            # Log un résumé rapide
            self._log_theme_summary(theme_id, theme_results["summary"])
        
        # Calculer les statistiques globales
        self.results["global_statistics"] = self._calculate_global_stats(all_articles_data)
        self.results["analysis_metadata"]["total_articles_analyzed"] = len(all_articles_data)
        
        # Sauvegarder les résultats
        await self._save_results()
        
    async def _analyze_theme(self, theme_id: str, theme_config: Dict) -> Dict:
        """Analyse une thématique spécifique."""
        config = WikipediaConfig(
            language='en',
            pages_per_category=theme_config.get('sample_size', 50),
            min_article_length=100,  # Très bas pour capturer tous les articles
            extract_infobox=True
        )
        
        articles_data = []

        # Stats de collecte pour debug
        collection_stats = {
            'attempted': 0,
            'succeeded': 0,
            'failed': 0,
            'rejected_quality': 0, 
            'rejected_filter': 0,  
            'errors': []
        }
        
        logger.info(f"Starting collection for {theme_id} with categories: {theme_config['categories']}")
        
        async with WikipediaCollector(config) as collector:
            # Override les catégories
            collector.PRIORITY_CATEGORIES = theme_config['categories']
            
            collected_count = 0
            errors_count = 0
            max_errors = 10  # Arrêter après trop d'erreurs
            
            async for doc in collector.extract():
                collection_stats['attempted'] += 1
                
                if collected_count >= int(theme_config.get('sample_size', 50)):
                    logger.info(f"✓ Reached target sample size: {collected_count}/{theme_config.get('sample_size', 50)}")
                    break
                
                if errors_count >= max_errors:
                    logger.warning(f"⚠️ Too many errors ({errors_count}), stopping collection for {theme_id}")
                    break
                
                try:
                    # Analyser l'article
                    article_analysis = self._analyze_article(doc)
                    article_analysis['theme'] = theme_id
                    articles_data.append(article_analysis)
                    
                    collected_count += 1
                    collection_stats['succeeded'] += 1
                    
                    # Log progress
                    if collected_count % 5 == 0:
                        logger.info(f"Progress: {collected_count}/{theme_config.get('sample_size', 50)} articles analyzed")

                        
                except Exception as e:
                    errors_count += 1
                    collection_stats['failed'] += 1
                    collection_stats['errors'].append(str(e))
                    logger.error(f"Error analyzing article: {e}")
                    continue
        
        # Calculer le résumé pour cette thématique
        summary = self._calculate_theme_summary(articles_data, theme_config)
        
        # Log final stats
        logger.info(f"""
        Collection stats for {theme_id}:
        - Attempted: {collection_stats['attempted']}
        - Succeeded: {collection_stats['succeeded']} 
        - Failed: {collection_stats['failed']}
        """)
        
        if collection_stats['errors']:
            logger.debug(f"Sample errors: {collection_stats['errors'][:3]}")

        if collection_stats['attempted'] > 0:
            rate = f"{collection_stats['succeeded'] / collection_stats['attempted'] * 100:.1f}%"
        else:
            rate = "0%"
        logger.info(f"Collection rate: {rate}")

        return {
            "articles": articles_data,
            "summary": summary
        }
    
    def _analyze_article(self, doc: Dict) -> Dict:
        """Analyse détaillée d'un article."""
        content = doc.get('content', {})
        metadata = doc.get('metadata', {})
        
        # Extraire le texte
        text = content.get('text_clean', '') or content.get('text', '')
        
        # Calculer le score de qualité
        quality_score, quality_components = self.scorer.score_document(doc)
        
        # Analyse structurelle
        sections = content.get('sections', {})
        has_gameplay_section = 'gameplay' in sections
        has_plot_section = any(s in sections for s in ['plot', 'story', 'synopsis'])
        has_development_section = 'development' in sections
        has_reception_section = any(s in sections for s in ['reception', 'critical_reception'])
        
        # Métadonnées du jeu
        game_info = metadata.get('game_info', {})
        
        # Analyse du texte
        words = text.split()
        sentences = text.split('.')
        
        # Gaming terms analysis
        gaming_terms_found = quality_components.get('gaming_terms_found', [])
        
        return {
            # Identification
            "title": content.get('title', 'Unknown'),
            "page_id": metadata.get('page_id', ''),
            "url": content.get('url', ''),
            
            # Métriques de taille
            "text_length": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            
            # Structure
            "section_count": len(sections),
            "has_infobox": bool(game_info),
            "has_gameplay_section": has_gameplay_section,
            "has_plot_section": has_plot_section,
            "has_development_section": has_development_section,
            "has_reception_section": has_reception_section,
            "sections_list": list(sections.keys()),
            
            # Qualité
            "quality_score": quality_score,
            "gaming_density": quality_components.get('gaming_density', 0),
            "relevance_score": quality_components.get('relevance', 0),
            "informativeness_score": quality_components.get('informativeness', 0),
            "length_score": quality_components.get('length', 0),
            
            # Gaming content
            "gaming_terms_count": len(gaming_terms_found),
            "gaming_terms_sample": gaming_terms_found[:10],
            
            # Métadonnées du jeu
            "release_year": game_info.get('release_year'),
            "developer": game_info.get('developer'),
            "publisher": game_info.get('publisher'),
            "genres": game_info.get('genre', ''),
            "platforms": game_info.get('platforms', ''),
            
            # Catégories Wikipedia
            "categories": metadata.get('categories', [])[:10]  # Top 10
        }
    
    def _calculate_theme_summary(self, articles: List[Dict], theme_config: Dict) -> Dict:
        """Calcule un résumé statistique pour une thématique."""
        if not articles:
            return self._empty_summary(theme_config)
        
        df = pd.DataFrame(articles)
        
        # Statistiques de base
        summary = {
            "theme_name": theme_config['name'],
            "description": theme_config.get('description', 'No description'),
            "article_count": len(articles),
            "target_count": theme_config.get('sample_size', 'Not specified'),
            
            # Longueur du texte
            "text_length": {
                "mean": int(df['text_length'].mean()),
                "median": int(df['text_length'].median()),
                "std": int(df['text_length'].std()),
                "min": int(df['text_length'].min()),
                "max": int(df['text_length'].max()),
                "percentiles": {
                    "25th": int(df['text_length'].quantile(0.25)),
                    "50th": int(df['text_length'].quantile(0.50)),
                    "75th": int(df['text_length'].quantile(0.75)),
                    "90th": int(df['text_length'].quantile(0.90))
                }
            },
            
            # Score de qualité
            "quality_score": {
                "mean": round(df['quality_score'].mean(), 3),
                "median": round(df['quality_score'].median(), 3),
                "std": round(df['quality_score'].std(), 3),
                "min": round(df['quality_score'].min(), 3),
                "max": round(df['quality_score'].max(), 3),
                "distribution": {
                    "excellent (>0.8)": len(df[df['quality_score'] > 0.8]),
                    "good (0.7-0.8)": len(df[(df['quality_score'] >= 0.7) & (df['quality_score'] <= 0.8)]),
                    "medium (0.6-0.7)": len(df[(df['quality_score'] >= 0.6) & (df['quality_score'] < 0.7)]),
                    "low (<0.6)": len(df[df['quality_score'] < 0.6])
                }
            },
            
            # Gaming density
            "gaming_density": {
                "mean": round(df['gaming_density'].mean(), 4),
                "median": round(df['gaming_density'].median(), 4),
                "std": round(df['gaming_density'].std(), 4),
                "min": round(df['gaming_density'].min(), 4),
                "max": round(df['gaming_density'].max(), 4)
            },
            
            # Structure
            "structure": {
                "avg_sections": round(df['section_count'].mean(), 1),
                "with_infobox": len(df[df['has_infobox'] == True]),
                "with_gameplay": len(df[df['has_gameplay_section'] == True]),
                "with_plot": len(df[df['has_plot_section'] == True]),
                "with_development": len(df[df['has_development_section'] == True]),
                "with_reception": len(df[df['has_reception_section'] == True])
            },
            
            # Articles par longueur
            "length_categories": {
                "very_short (<500)": len(df[df['text_length'] < 500]),
                "short (500-1000)": len(df[(df['text_length'] >= 500) & (df['text_length'] < 1000)]),
                "medium (1000-5000)": len(df[(df['text_length'] >= 1000) & (df['text_length'] < 5000)]),
                "long (5000-10000)": len(df[(df['text_length'] >= 5000) & (df['text_length'] < 10000)]),
                "very_long (>10000)": len(df[df['text_length'] >= 10000])
            },
            
            # Top gaming terms
            "top_gaming_terms": self._get_top_gaming_terms(articles),
            
            # Années de sortie (si disponible)
            "release_years": self._analyze_release_years(df)
        }
        
        return summary
    
    def _get_top_gaming_terms(self, articles: List[Dict]) -> List[Tuple[str, int]]:
        """Extrait les termes gaming les plus fréquents."""
        term_counts = defaultdict(int)
        
        for article in articles:
            for term in article.get('gaming_terms_sample', []):
                term_counts[term] += 1
        
        # Top 15 termes
        return sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    def _analyze_release_years(self, df: pd.DataFrame) -> Dict:
        """Analyse la distribution des années de sortie."""
        years = df['release_year'].dropna()
        
        if len(years) == 0:
            return {"available_data": False}
        
        return {
            "available_data": True,
            "count_with_year": len(years),
            "min_year": int(years.min()) if len(years) > 0 else None,
            "max_year": int(years.max()) if len(years) > 0 else None,
            "median_year": int(years.median()) if len(years) > 0 else None
        }
    
    def _calculate_global_stats(self, all_articles: List[Dict]) -> Dict:
        """Calcule des statistiques globales sur tous les articles."""
        if not all_articles:
            return {}
        
        df = pd.DataFrame(all_articles)
        
        return {
            "total_articles": len(all_articles),
            "themes_analyzed": len(self.THEMES),
            
            # Comparaison entre thèmes
            "quality_by_theme": {
                theme: {
                        "mean_quality": round(df[df['theme'] == theme]['quality_score'].mean(), 3) if not df[df['theme'] == theme]['quality_score'].isna().all() else 0,
                        "mean_length": int(df[df['theme'] == theme]['text_length'].mean()) if not df[df['theme'] == theme]['text_length'].isna().all() else 0,
                        "mean_gaming_density": round(df[df['theme'] == theme]['gaming_density'].mean(), 4) if not df[df['theme'] == theme]['gaming_density'].isna().all() else 0
                        }
                for theme in self.THEMES.keys()
            },
            
            # Corrélations
            "correlations": {
                "length_vs_quality": round(df['text_length'].corr(df['quality_score']), 3),
                "sections_vs_quality": round(df['section_count'].corr(df['quality_score']), 3),
                "gaming_density_vs_quality": round(df['gaming_density'].corr(df['quality_score']), 3)
            },
            
            # Recommandations basées sur l'analyse
            "recommendations": self._generate_recommendations(df)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []
        
        # Analyser la qualité moyenne par thème
        theme_quality = df.groupby('theme')['quality_score'].mean().sort_values(ascending=False)
        
        # Meilleures thématiques
        best_themes = theme_quality.head(3).index.tolist()
        recommendations.append(
            f"Prioriser les thématiques: {', '.join([self.THEMES[t]['name'] for t in best_themes])}"
        )
        
        # Seuil de longueur optimal
        length_bins = pd.cut(df['text_length'], bins=[0, 1000, 5000, 10000, float('inf')])
        quality_by_length = df.groupby(length_bins)['quality_score'].mean()
        optimal_length = quality_by_length.idxmax()
        recommendations.append(
            f"Longueur optimale des articles: {optimal_length} caractères"
        )
        
        # Structure importante
        structure_impact = {
            'infobox': df[df['has_infobox']]['quality_score'].mean() - df[~df['has_infobox']]['quality_score'].mean(),
            'gameplay': df[df['has_gameplay_section']]['quality_score'].mean() - df[~df['has_gameplay_section']]['quality_score'].mean(),
            'reception': df[df['has_reception_section']]['quality_score'].mean() - df[~df['has_reception_section']]['quality_score'].mean()
        }
        
        most_important = max(structure_impact.items(), key=lambda x: x[1])
        recommendations.append(
            f"Prioriser les articles avec {most_important[0]} (+{most_important[1]:.3f} qualité)"
        )
        
        return recommendations
    
    def _empty_summary(self, theme_config: Dict) -> Dict:
        """Retourne un résumé vide si pas de données."""
        return {
            "theme_name": theme_config['name'],
            "description": theme_config.get('description', 'No description'),
            "article_count": 0,
            "error": "No articles collected"
        }
    
    def _log_theme_summary(self, theme_id: str, summary: Dict):
        """Affiche un résumé rapide dans les logs."""
        logger.info(f"\n📊 Summary for {summary['theme_name']}:")
        logger.info(f"   Articles analyzed: {summary.get('article_count', 0)}")
        if 'text_length' in summary:
            logger.info(f"   Avg text length: {summary['text_length']['mean']:,} chars")
        if 'quality_score' in summary:
            logger.info(f"   Avg quality score: {summary['quality_score']['mean']:.3f}")
        if 'structure' in summary:
            logger.info(f"   With infobox: {summary['structure']['with_infobox']}/{summary['article_count']}")
        
    async def _save_results(self):
        """Sauvegarde les résultats dans plusieurs formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON complet
        json_path = self.output_dir / f"wikipedia_quality_analysis_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved complete results to {json_path}")
        
        # 2. CSV summary par thème
        summary_data = []
        for theme_id, summary in self.results['summary_by_theme'].items():
            summary_data.append({
                'theme': theme_id,
                'name': summary['theme_name'],
                'articles': summary['article_count'],
                'avg_length': summary.get('text_length', {}).get('mean', 0),
                'avg_quality': summary.get('quality_score', {}).get('mean', 0),
                'avg_gaming_density': summary.get('gaming_density', {}).get('mean', 0),
                'with_infobox_%': summary.get('structure', {}).get('with_infobox', 0) / summary.get('article_count', 1) * 100 if summary.get('article_count', 0) > 0 else 0,
                'excellent_quality_%': summary.get('quality_score', {}).get('distribution', {}).get('excellent (>0.8)', 0) / summary.get('article_count', 1) * 100 if summary.get('article_count', 0) > 0 else 0            
                })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / f"wikipedia_quality_summary_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary CSV to {csv_path}")
        
        # 3. Rapport lisible en Markdown
        report_path = self.output_dir / f"wikipedia_quality_report_{timestamp}.md"
        self._generate_markdown_report(report_path)
        logger.info(f"Saved readable report to {report_path}")
    
    def _generate_markdown_report(self, path: Path):
        """Génère un rapport Markdown lisible."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Wikipedia Gaming Articles Quality Analysis\n\n")
            f.write(f"Generated: {self.results['analysis_metadata']['timestamp']}\n\n")
            f.write(f"Total articles analyzed: {self.results['analysis_metadata']['total_articles_analyzed']}\n\n")
            
            # Résumé par thème
            f.write("## Summary by Theme\n\n")
            for theme_id, summary in self.results['summary_by_theme'].items():
                f.write(f"### {summary['theme_name']}\n")
                f.write(f"*{summary['description']}*\n\n")
                f.write(f"- Articles analyzed: {summary['article_count']}\n")
                if 'text_length' in summary:
                    f.write(f"- Average length: {summary['text_length']['mean']:,} chars\n")
                if 'quality_score' in summary:
                    f.write(f"- Average quality: {summary['quality_score']['mean']:.3f}\n")
                if 'gaming_density' in summary:
                    f.write(f"- Gaming density: {summary['gaming_density']['mean']:.4f}\n")
                if 'structure' in summary and 'article_count' in summary:
                    f.write(f"- With infobox: {summary['structure']['with_infobox']}/{summary['article_count']}\n")
                f.write("\n")
            
            # Statistiques globales
            if 'global_statistics' in self.results:
                f.write("## Global Statistics\n\n")
                
                # Top thèmes par qualité
                f.write("### Best themes by quality:\n")
                quality_by_theme = self.results['global_statistics']['quality_by_theme']
                sorted_themes = sorted(quality_by_theme.items(), 
                                     key=lambda x: x[1]['mean_quality'], 
                                     reverse=True)
                for theme, stats in sorted_themes[:5]:
                    f.write(f"1. {self.THEMES[theme]['name']}: {stats['mean_quality']:.3f}\n")
                
                f.write("\n### Recommendations:\n")
                for rec in self.results['global_statistics']['recommendations']:
                    f.write(f"- {rec}\n")


async def main():
    """Fonction principale pour lancer l'analyse."""
    output_dir = Path("data/wikipedia_quality_analysis")
    analyzer = WikipediaQualityAnalyzer(output_dir)
    
    await analyzer.analyze_all_themes()
    
    logger.info("\n✅ Analysis complete! Check the output directory for results.")


if __name__ == "__main__":
    asyncio.run(main())


THEMES1 = {
    "Featured_Articles": {
        "name": "Articles de qualité",
        "categories": [
            "Category:Featured_video_games_articles",
            "Category:GA-Class_video_game_articles",
            "Category:Good_video_game_articles"
        ],
        "sample_size": 100,
        "description": "Articles de haute qualité éditoriale sur des jeux ou sujets connexes"
    },
    "Recent_Games": {
        "name": "Jeux récents (2022-2024)",
        "categories": [
            "Category:2024_video_games",
            "Category:2023_video_games",
            "Category:2022_video_games"
        ],
        "sample_size": 100,
        "description": "Jeux récents, avec contenu en développement mais d'actualité"
    },
    "Game_Mechanics": {
        "name": "Mécaniques de jeu",
        "categories": [
            "Category:Video_game_gameplay",
            "Category:Video_game_mechanics",
            "Category:Video_game_terminology"
        ],
        "sample_size": 50,
        "description": "Articles expliquant les systèmes, boucles de gameplay et concepts techniques"
    },
    "Franchises": {
        "name": "Franchises et séries",
        "categories": [
            "Category:Video_game_franchises",
            "Category:Long-running_video_game_franchises",
            "Category:Video_game_sequels"
        ],
        "sample_size": 50,
        "description": "Univers riches avec lore, personnages récurrents et évolution historique"
    },
    "Esports": {
        "name": "Esports et compétitif",
        "categories": [
            "Category:Esports_games",
            "Category:Fighting_games",
            "Category:Multiplayer_online_battle_arena_games"
        ],
        "sample_size": 50,
        "description": "Jeux compétitifs avec vocabulaire et méta spécifique"
    },
    "Indies": {
        "name": "Jeux indépendants",
        "categories": [
            "Category:Indie_games",
            "Category:Independent_video_games",
            "Category:IGF_Grand_Prize_winners"
        ],
        "sample_size": 50,
        "description": "Jeux innovants, souvent à fort contenu narratif ou expérimental"
    },
    "Retro": {
        "name": "Jeux rétro",
        "categories": [
            "Category:1980s_video_games",
            "Category:1990s_video_games",
            "Category:Arcade_video_games"
        ],
        "sample_size": 50,
        "description": "Fondations historiques du jeu vidéo et icônes du passé"
    },
    "Genres": {
        "name": "Genres vidéoludiques",
        "categories": [
            "Category:Video_game_genres"
        ],
        "sample_size": 50,
        "description": "Typologie des jeux (RPG, FPS, simulation...) et leurs spécificités linguistiques"
    },
    "Notable_Developers": {
        "name": "Studios emblématiques",
        "categories": [
            "Category:Video_game_development_companies",
            "Category:Video_game_publishers"
        ],
        "sample_size": 50,
        "description": "Connaissances sur les créateurs majeurs de jeux, leurs tendances et innovations"
    },
    "Development": {
        "name": "Développement de jeux",
        "categories": [
            "Category:Video_game_development",
            "Category:Video_game_design",
            "Category:Video_game_programming"
        ],
        "sample_size": 50,
        "description": "Articles techniques sur la conception, le design et le développement"
    },
    "Game_Engines": {
        "name": "Moteurs de jeu",
        "categories": [
            "Category:Video_game_engines"
        ],
        "sample_size": 30,
        "description": "Articles sur les technologies sous-jacentes aux jeux (Unity, Unreal...)"
    },
    "Gaming_Culture": {
        "name": "Culture vidéoludique",
        "categories": [
            "Category:Video_game_culture",
            "Category:Gaming_terminology",
            "Category:Let's_Plays"
        ],
        "sample_size": 50,
        "description": "Mèmes, pratiques communautaires, streaming, etc."
    },
    "Narrative_Design": {
        "name": "Narration dans le jeu vidéo",
        "categories": [
            "Category:Narrative_video_games",
            "Category:Interactive_storytelling"
        ],
        "sample_size": 30,
        "description": "Articles sur la construction narrative, choix du joueur, etc."
    },
    "Music_And_Sound": {
        "name": "Musique et son",
        "categories": [
            "Category:Video_game_music",
            "Category:Video_game_composers"
        ],
        "sample_size": 30,
        "description": "Importance sonore dans l’immersion et la reconnaissance des jeux"
    },
    "AI_And_Behavior": {
        "name": "IA et comportement dans les jeux",
        "categories": [
            "Category:Artificial_intelligence_in_video_games",
            "Category:Video_game_bots"
        ],
        "sample_size": 30,
        "description": "Articles sur l’IA, pathfinding, comportements des NPC"
    },
    "Game_Lore_And_Universe": {
        "name": "Univers et lore de jeux",
        "categories": [
            "Category:Video_game_lore",
            "Category:Video_game_characters",
            "Category:Fictional_universes"
        ],
        "sample_size": 50,
        "description": "Articles sur les univers fictionnels profonds et les personnages majeurs"
    },
    "Multiplayer_And_Online": {
        "name": "Jeux multijoueurs et en ligne",
        "categories": [
            "Category:Massively_multiplayer_online_games",
            "Category:Online_video_games"
        ],
        "sample_size": 50,
        "description": "Vocabulaire, dynamiques communautaires et techniques spécifiques"
    },
    "Modding_And_Communities": {
        "name": "Modding et communautés",
        "categories": [
            "Category:Video_game_modding",
            "Category:Video_game_fandom"
        ],
        "sample_size": 30,
        "description": "Articles sur la création de mods, les communautés de fans, etc."
    },
    "Speedrunning": {
        "name": "Speedrunning",
        "categories": [
            "Category:Speedrun",
            "Category:Speedrunning_techniques"
        ],
        "sample_size": 30,
        "description": "Langage et pratiques spécifiques au speedrunning"
    },
    "Awards_And_Recognition": {
        "name": "Récompenses vidéoludiques",
        "categories": [
            "Category:Video_game_awards",
            "Category:The_Game_Awards"
        ],
        "sample_size": 30,
        "description": "Jeux reconnus pour leur qualité ou innovation"
    }
}

THEMES2 = {
        "Platform_Exclusives": {
            "name": "Exclusivités plateformes",
            "categories": [
                "Category:PlayStation_exclusive_games",
                "Category:Nintendo_Switch_exclusive_games",
                "Category:Xbox_exclusive_games"
            ],
            "sample_size": 75,
            "description": "Jeux exclusifs à une plateforme, souvent très documentés et discutés"
        },
        "Horror_Games": {
            "name": "Jeux d'horreur",
            "categories": [
                "Category:Horror_video_games",
                "Category:Survival_horror_games",
                "Category:Psychological_horror_games"
            ],
            "sample_size": 50,
            "description": "Genre avec vocabulaire spécifique sur l'atmosphère, tension et mécaniques de peur"
        },
        "Open_World": {
            "name": "Mondes ouverts",
            "categories": [
                "Category:Open-world_video_games",
                "Category:Sandbox_video_games",
                "Category:Nonlinear_gameplay"
            ],
            "sample_size": 75,
            "description": "Jeux avec exploration libre, quêtes secondaires et systèmes émergents"
        },
        "Strategy_Games": {
            "name": "Jeux de stratégie",
            "categories": [
                "Category:Strategy_video_games",
                "Category:Real-time_strategy_video_games",
                "Category:Turn-based_strategy_video_games"
            ],
            "sample_size": 60,
            "description": "Vocabulaire tactique, gestion de ressources et mécaniques complexes"
        },
        "Educational_Games": {
            "name": "Jeux éducatifs",
            "categories": [
                "Category:Educational_video_games",
                "Category:Children's_educational_video_games",
                "Category:Serious_games"
            ],
            "sample_size": 50,
            "description": "Jeux conçus pour l'apprentissage avec mécaniques pédagogiques"
        },
        "Remakes_Remasters": {
            "name": "Remakes et remasters",
            "categories": [
                "Category:Video_game_remakes",
                "Category:Video_game_remasters",
                "Category:HD_remasters"
            ],
            "sample_size": 50,
            "description": "Comparaisons entre versions, améliorations techniques et nostalgies"
        },
        "Mobile_Gaming": {
            "name": "Jeux mobiles",
            "categories": [
                "Category:Mobile_games",
                "Category:Android_games",
                "Category:IOS_games"
            ],
            "sample_size": 60,
            "description": "Écosystème mobile avec monétisation F2P et mécaniques tactiles"
        },
        "VR_AR_Games": {
            "name": "Réalité virtuelle et augmentée",
            "categories": [
                "Category:Virtual_reality_games",
                "Category:Oculus_Quest_games",
                "Category:PlayStation_VR_games"
            ],
            "sample_size": 50,
            "description": "Technologies immersives avec vocabulaire spécifique d'interaction"
        },
        "Sports_Racing": {
            "name": "Sports et course",
            "categories": [
                "Category:Sports_video_games",
                "Category:Racing_video_games",
                "Category:Simulation_racing_games"
            ],
            "sample_size": 50,
            "description": "Simulations sportives et courses avec terminologie technique"
        },
        "Japanese_Games": {
            "name": "Jeux japonais",
            "categories": [
                "Category:Japan-exclusive_video_games",
                "Category:Japanese_role-playing_video_games",
                "Category:Visual_novels"
            ],
            "sample_size": 60,
            "description": "Productions japonaises avec conventions narratives et gameplay distincts"
        },
        "Puzzle_Games": {
            "name": "Jeux de puzzle",
            "categories": [
                "Category:Puzzle_video_games",
                "Category:Logic_puzzle_video_games",
                "Category:Physics-based_puzzle_games"
            ],
            "sample_size": 50,
            "description": "Mécaniques de résolution de problèmes et design de niveaux"
        },
        "Controversial_Games": {
            "name": "Jeux controversés",
            "categories": [
                "Category:Obscenity_controversies_in_video_games",
                "Category:Censored_video_games",
                "Category:Video_game_controversies"
            ],
            "sample_size": 50,
            "description": "Jeux ayant suscité débats sur violence, contenu ou mécaniques"
        },
        "Game_Series_Crossovers": {
            "name": "Crossovers et collaborations",
            "categories": [
                "Category:Video_game_crossovers",
                "Category:Crossover_video_games",
                "Category:Video_game_guest_characters"
            ],
            "sample_size": 50,
            "description": "Mélanges d'univers et mécaniques de franchises différentes"
        },
        "Roguelike_Roguelite": {
            "name": "Roguelike et roguelite",
            "categories": [
                "Category:Roguelike_video_games",
                "Category:Roguelite_video_games",
                "Category:Procedural_generation_in_video_games"
            ],
            "sample_size": 50,
            "description": "Génération procédurale, permadeath et progression meta"
        },
        "Simulation_Games": {
            "name": "Jeux de simulation",
            "categories": [
                "Category:Simulation_video_games",
                "Category:Life_simulation_games",
                "Category:Business_simulation_games"
            ],
            "sample_size": 50,
            "description": "Reproduction de systèmes réels avec mécaniques détaillées"
        },
        "Discontinued_Cancelled": {
            "name": "Jeux annulés et abandonnés",
            "categories": [
                "Category:Cancelled_video_games",
                "Category:Vaporware_video_games",
                "Category:Unreleased_video_games"
            ],
            "sample_size": 50,
            "description": "Projets avortés offrant insights sur le développement"
        },
        "Game_Preservation": {
            "name": "Préservation du jeu vidéo",
            "categories": [
                "Category:Video_game_preservation",
                "Category:Abandonware_games",
                "Category:Video_game_emulation"
            ],
            "sample_size": 50,
            "description": "Conservation du patrimoine ludique et accès aux œuvres anciennes"
        },
        "Accessibility_Gaming": {
            "name": "Accessibilité dans le jeu",
            "categories": [
                "Category:Accessible_video_games",
                "Category:Video_game_accessibility",
                "Category:Games_for_visually_impaired_players"
            ],
            "sample_size": 50,
            "description": "Options d'accessibilité et design inclusif"
        },
        "Game_Journalism": {
            "name": "Journalisme vidéoludique",
            "categories": [
                "Category:Video_game_journalism",
                "Category:Video_game_critics",
                "Category:Video_game_websites"
            ],
            "sample_size": 50,
            "description": "Médias, critiques et analyse du jeu vidéo"
        },
        "Experimental_Art_Games": {
            "name": "Jeux expérimentaux et artistiques",
            "categories": [
                "Category:Art_games",
                "Category:Experimental_video_games",
                "Category:Indie_game_award_winners"
            ],
            "sample_size": 50,
            "description": "Jeux repoussant les limites du médium avec approches avant-gardistes"
        }
    }
