"""
Setup script for gaming dataset collector.
"""

from setuptools import setup, find_packages

setup(
    name="gaming-dataset-collector",
    version="1.0.0",
    description="A comprehensive collector for gaming-related data to train LLMs",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core
        "pandas>=2.0.3",
        "pyarrow>=12.0.1",
        "numpy>=1.24.3",
        
        # Async & Networking
        "aiohttp>=3.8.5",
        "aiofiles>=23.1.0",
        "requests>=2.31.0",
        
        # NLP & Processing  
        "spacy>=3.6.0",
        "beautifulsoup4>=4.12.2",
        "lxml>=4.9.3",
        "ftfy>=6.1.1",
        
        # APIs
        "praw>=7.7.1",
        "google-api-python-client>=2.95.0",
        "mwclient>=0.10.1",
        
        # Storage
        "boto3>=1.28.17",
        "minio>=7.1.16",
        
        # Utilities
        "python-dotenv>=1.0.0",
        "click>=8.1.6",
        "tqdm>=4.65.0",
        "tenacity>=8.2.2",
        "ratelimit>=2.2.1",
        
        # Logging
        "loguru>=0.7.0",
        
        # Data Quality
        "great-expectations>=0.17.12",
        "jsonschema>=4.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "collect-wikipedia=collectors.wikipedia_collector:main",
            "collect-steam=collectors.steam_collector:main",
        ]
    },
)