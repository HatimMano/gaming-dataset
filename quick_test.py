import asyncio
from collectors.wikipedia_collector import WikipediaCollector, WikipediaConfig
from processors.quality_scorer import EnhancedQualityScorer

async def test_collector():
    config = WikipediaConfig(
        pages_per_category=5,  # Seulement 5 pages
        categories_per_batch=1  # Une seule catégorie
    )
    
    collector = WikipediaCollector(config)
    scorer = EnhancedQualityScorer()
    
    # Override pour tester une seule catégorie
    collector.PRIORITY_CATEGORIES = ["Category:2024_video_games"]
    
    count = 0
    async with collector:
        async for doc in collector.extract():
            count += 1
            score, components = scorer.score_document(doc)
            print(f"\n{'='*50}")
            print(f"Article {count}: {doc['content']['title']}")
            print(f"Score: {score:.3f}")
            print(f"Length: {len(doc['content'].get('text', ''))} chars")
            print(f"Gaming density: {components.get('gaming_density', 0):.3f}")
            
            if count >= 3:  # Stop après 3 articles
                break
    
    print(f"\nTest completed! Processed {count} articles")

if __name__ == "__main__":
    asyncio.run(test_collector())