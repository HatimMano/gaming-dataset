"""
Parquet storage manager for efficient data storage and retrieval.
Handles schema validation, compression, and partitioning.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from dataclasses import asdict
from datetime import datetime


class ParquetManager:
    """Manage parquet file storage for the gaming dataset."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize the parquet manager.
        
        Args:
            base_path: Base directory for storing parquet files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.by_source_path = self.base_path / "by_source"
        self.by_date_path = self.base_path / "by_date"
        self.merged_path = self.base_path / "merged"
        
        for path in [self.by_source_path, self.by_date_path, self.merged_path]:
            path.mkdir(exist_ok=True)
        
        # Define comprehensive parquet schema
        self.schema = self._create_schema()
        
        # Compression settings
        self.compression = 'snappy'  # Good balance of speed and size
        self.compression_level = None  # Default for snappy
        
        logger.info(f"Initialized ParquetManager at {self.base_path}")
    
    def _create_schema(self) -> pa.Schema:
        """Create the PyArrow schema for consistent data structure."""
        return pa.schema([
            # Document identification
            ('document_id', pa.string()),
            
            # Source information
            ('source', pa.struct([
                ('platform', pa.string()),
                ('url', pa.string()),
                ('api_endpoint', pa.string()),
                ('crawl_timestamp', pa.string())
            ])),
            
            # Content
            ('content', pa.struct([
                ('title', pa.string()),
                ('text', pa.large_string()),  # Use large_string for long texts
                ('text_clean', pa.large_string()),
                ('language', pa.string()),
                ('word_count', pa.int32()),
                ('char_count', pa.int32())
            ])),
            
            # Metadata
            ('metadata', pa.struct([
                # Game information
                ('game', pa.struct([
                    ('title', pa.string()),
                    ('aliases', pa.list_(pa.string())),
                    ('game_id', pa.string()),
                    ('release_date', pa.string()),
                    ('developer', pa.string()),
                    ('publisher', pa.string()),
                    ('genres', pa.list_(pa.string())),
                    ('platforms', pa.list_(pa.string())),
                    ('tags', pa.list_(pa.string()))
                ])),
                
                # Classification
                ('classification', pa.struct([
                    ('content_type', pa.string()),
                    ('genres', pa.list_(pa.string())),
                    ('platforms', pa.list_(pa.string())),
                    ('tags', pa.list_(pa.string())),
                    ('is_tutorial', pa.bool_()),
                    ('is_review', pa.bool_()),
                    ('is_news', pa.bool_())
                ])),
                
                # Author information
                ('author', pa.struct([
                    ('name', pa.string()),
                    ('id', pa.string()),
                    ('verified', pa.bool_()),
                    ('platform', pa.string())
                ]))
            ])),
            
            # Quality scores
            ('quality', pa.struct([
                ('overall', pa.float32()),
                ('relevance', pa.float32()),
                ('completeness', pa.float32()),
                ('uniqueness', pa.float32()),
                ('gaming_density', pa.float32()),
                ('structure', pa.float32()),
                ('freshness', pa.float32())
            ])),
            
            # Processing information
            ('processing', pa.struct([
                ('pipeline_version', pa.string()),
                ('extracted_entities', pa.list_(pa.string())),
                ('sentiment', pa.string()),
                ('toxicity_score', pa.float32()),
                ('processed_at', pa.string())
            ]))
        ])
    
    def save(self, df: pd.DataFrame, filename: str, partition_by_source: bool = True) -> int:
        """Save DataFrame to parquet file with optional partitioning.
        
        Args:
            df: DataFrame to save
            filename: Name of the parquet file
            partition_by_source: Whether to save in source-specific directory
            
        Returns:
            File size in bytes
        """
        try:
            # Determine save path
            if partition_by_source and len(df) > 0:
                # Get the most common source
                source = df['source'].apply(lambda x: x.get('platform', 'unknown') if isinstance(x, dict) else x.platform).mode()[0]
                filepath = self.by_source_path / source / filename
                filepath.parent.mkdir(exist_ok=True)
            else:
                filepath = self.base_path / filename
            
            # Also save by date
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_filepath = self.by_date_path / date_str / filename
            date_filepath.parent.mkdir(exist_ok=True)
            
            # Validate and convert to PyArrow table
            table = self._dataframe_to_table(df)
            
            # Write with compression and metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'record_count': str(len(df)),
                'schema_version': '1.0',
                'compression': self.compression
            }
            
            pq.write_table(
                table,
                filepath,
                compression=self.compression,
                compression_level=self.compression_level,
                use_dictionary=True,
            )
            
            # Also save to date partition
            pq.write_table(
                table,
                date_filepath,
                compression=self.compression,
                compression_level=self.compression_level,
                use_dictionary=True,
            )
            
            file_size = filepath.stat().st_size
            logger.info(
                f"Saved {len(df)} records to {filepath.relative_to(self.base_path)} "
                f"({self._format_size(file_size)})"
            )
            
            # Save metadata
            self._save_metadata(filepath, df)
            
            return file_size
            
        except Exception as e:
            logger.error(f"Error saving parquet file: {e}")
            raise
    
    
    def _dataframe_to_table(self, df: pd.DataFrame) -> pa.Table:
        """Convert DataFrame to PyArrow Table with schema validation."""
        try:
            # Créer une copie pour ne pas modifier l'original
            df_copy = df.copy()
            
            # Fonction de conversion récursive
            def convert_value(obj):
                if obj is None:
                    return None
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, '__dataclass_fields__'):
                    result = {}
                    for field in obj.__dataclass_fields__:
                        value = getattr(obj, field)
                        result[field] = convert_value(value)  # Récursif
                    return result
                elif isinstance(obj, dict):
                    return {k: convert_value(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_value(item) for item in obj]
                else:
                    return obj
            
            # Appliquer la conversion à toutes les colonnes
            for col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(convert_value)
            
            # Convertir en PyArrow table
            table = pa.Table.from_pandas(df_copy, schema=self.schema)
            return table
            
        except Exception as e:
            logger.error(f"Error converting DataFrame to Table: {e}")
            # Try without schema for debugging
            table = pa.Table.from_pandas(df_copy)
            logger.info(f"Table schema without enforcement: {table.schema}")
            raise


    def load(self, filename: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load parquet file into DataFrame.
        
        Args:
            filename: Name or path of the parquet file
            columns: Specific columns to load (None for all)
            
        Returns:
            DataFrame with loaded data
        """
        # Try multiple locations
        possible_paths = [
            self.base_path / filename,
            self.by_source_path / "**" / filename,
            self.by_date_path / "**" / filename,
            self.merged_path / filename
        ]
        
        filepath = None
        for path_pattern in possible_paths:
            if "*" in str(path_pattern):
                matches = list(self.base_path.glob(str(Path(path_pattern).relative_to(self.base_path))))
                if matches:
                    filepath = matches[0]
                    break
            elif path_pattern.exists():
                filepath = path_pattern
                break
        
        if not filepath:
            raise FileNotFoundError(f"Parquet file not found: {filename}")
        
        df = pd.read_parquet(filepath, columns=columns)
        logger.info(f"Loaded {len(df)} records from {filepath.relative_to(self.base_path)}")
        
        return df
    
    def list_files(self, pattern: str = "*.parquet") -> List[Dict[str, Any]]:
        """List all parquet files with metadata.
        
        Args:
            pattern: Glob pattern for filtering files
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        for filepath in self.base_path.rglob(pattern):
            if filepath.is_file():
                stat = filepath.stat()
                
                # Try to get record count from metadata
                record_count = None
                try:
                    meta_path = filepath.with_suffix('.meta.json')
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            record_count = meta.get('record_count')
                except:
                    pass
                
                files.append({
                    'filename': filepath.name,
                    'path': str(filepath.relative_to(self.base_path)),
                    'size': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'record_count': record_count
                })
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def merge_files(
        self, 
        output_filename: str, 
        input_pattern: str = "*.parquet",
        delete_originals: bool = False,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """Merge multiple parquet files into a single file.
        
        Args:
            output_filename: Name for the merged file
            input_pattern: Pattern for files to merge
            delete_originals: Whether to delete original files after merging
            deduplicate: Whether to remove duplicate documents
            
        Returns:
            Dictionary with merge statistics
        """
        files = list(self.base_path.rglob(input_pattern))
        
        # Exclude the output file if it exists
        output_path = self.merged_path / output_filename
        files = [f for f in files if f != output_path]
        
        if not files:
            logger.warning("No files to merge")
            return {'merged_files': 0, 'total_records': 0}
        
        logger.info(f"Merging {len(files)} files...")
        
        # Read all files
        dfs = []
        total_records = 0
        
        for filepath in files:
            try:
                df = pd.read_parquet(filepath)
                total_records += len(df)
                dfs.append(df)
                logger.debug(f"Read {len(df)} records from {filepath.name}")
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
                continue
        
        if not dfs:
            logger.error("No data frames to merge")
            return {'merged_files': 0, 'total_records': 0}
        
        # Merge
        merged_df = pd.concat(dfs, ignore_index=True)
        original_count = len(merged_df)
        
        # Deduplicate if requested
        duplicates_removed = 0
        if deduplicate:
            merged_df = merged_df.drop_duplicates(subset=['document_id'], keep='last')
            duplicates_removed = original_count - len(merged_df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate documents")
        
        # Save merged file
        self.save(merged_df, output_filename, partition_by_source=False)
        
        # Delete originals if requested
        if delete_originals:
            for filepath in files:
                try:
                    filepath.unlink()
                    # Also delete metadata
                    meta_path = filepath.with_suffix('.meta.json')
                    if meta_path.exists():
                        meta_path.unlink()
                    logger.debug(f"Deleted {filepath.name}")
                except Exception as e:
                    logger.error(f"Error deleting {filepath}: {e}")
        
        return {
            'merged_files': len(files),
            'total_records': len(merged_df),
            'original_records': total_records,
            'duplicates_removed': duplicates_removed,
            'output_file': str(output_path.relative_to(self.base_path)),
            'output_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored data.
        
        Returns:
            Dictionary with detailed statistics
        """
        all_files = self.list_files()
        
        if not all_files:
            return {
                'file_count': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'sources': {},
                'content_types': {},
                'avg_document_size_kb': 0
            }
        
        # Aggregate statistics
        total_size = sum(f['size'] for f in all_files)
        total_records = 0
        sources = {}
        content_types = {}
        quality_scores = []
        
        # Sample files for detailed stats (don't load everything)
        sample_size = min(10, len(all_files))
        sample_files = all_files[:sample_size]
        
        for file_info in sample_files:
            try:
                df = self.load(file_info['filename'])
                total_records += len(df)
                
                # Count sources
                if 'source' in df.columns:
                    for source in df['source'].apply(lambda x: x.get('platform', 'unknown') if isinstance(x, dict) else 'unknown'):
                        sources[source] = sources.get(source, 0) + 1
                
                # Count content types
                if 'metadata' in df.columns:
                    for _, row in df.iterrows():
                        if isinstance(row['metadata'], dict):
                            ct = row['metadata'].get('classification', {}).get('content_type', 'unknown')
                            content_types[ct] = content_types.get(ct, 0) + 1
                
                # Collect quality scores
                if 'quality' in df.columns:
                    for _, row in df.iterrows():
                        if isinstance(row['quality'], dict):
                            score = row['quality'].get('overall', 0)
                            if score > 0:
                                quality_scores.append(score)
                                
            except Exception as e:
                logger.error(f"Error processing {file_info['filename']} for stats: {e}")
                continue
        
        # Extrapolate if we only sampled
        if sample_size < len(all_files):
            multiplier = len(all_files) / sample_size
            total_records = int(total_records * multiplier)
            sources = {k: int(v * multiplier) for k, v in sources.items()}
            content_types = {k: int(v * multiplier) for k, v in content_types.items()}
        
        # Calculate averages
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_doc_size = (total_size / total_records / 1024) if total_records > 0 else 0
        
        return {
            'file_count': len(all_files),
            'total_records': total_records,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'sources': dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)),
            'content_types': dict(sorted(content_types.items(), key=lambda x: x[1], reverse=True)),
            'avg_quality_score': round(avg_quality, 3),
            'avg_document_size_kb': round(avg_doc_size, 2),
            'latest_file': all_files[0]['filename'] if all_files else None,
            'oldest_file': all_files[-1]['filename'] if all_files else None
        }
    
    def query(
        self, 
        filters: Dict[str, Any],
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Query parquet files with filters.
        
        Args:
            filters: Dictionary of filters to apply
            columns: Columns to return
            limit: Maximum number of records
            
        Returns:
            Filtered DataFrame
        """
        # This is a simple implementation - could be optimized with predicate pushdown
        all_files = self.list_files()
        results = []
        records_found = 0
        
        for file_info in all_files:
            if limit and records_found >= limit:
                break
                
            try:
                df = self.load(file_info['filename'], columns=columns)
                
                # Apply filters
                mask = pd.Series([True] * len(df))
                
                for field, value in filters.items():
                    if '.' in field:  # Nested field
                        parts = field.split('.')
                        if parts[0] in df.columns:
                            mask &= df[parts[0]].apply(
                                lambda x: x.get(parts[1]) == value if isinstance(x, dict) else False
                            )
                    else:
                        if field in df.columns:
                            mask &= df[field] == value
                
                filtered_df = df[mask]
                
                if len(filtered_df) > 0:
                    if limit:
                        remaining = limit - records_found
                        filtered_df = filtered_df.head(remaining)
                    
                    results.append(filtered_df)
                    records_found += len(filtered_df)
                    
            except Exception as e:
                logger.error(f"Error querying {file_info['filename']}: {e}")
                continue
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _save_metadata(self, filepath: Path, df: pd.DataFrame):
        """Save metadata file alongside parquet file."""
        meta = {
            'created_at': datetime.now().isoformat(),
            'record_count': len(df),
            'file_size': filepath.stat().st_size,
            'sources': df['source'].apply(lambda x: x.get('platform', 'unknown') if isinstance(x, dict) else 'unknown').value_counts().to_dict() if 'source' in df.columns else {},
            'quality_stats': {
                'mean': 0,
                'min': 0,
                'max': 0
            }
        }
        
        # Calculate quality stats if available
        if 'quality' in df.columns:
            quality_scores = []
            for _, row in df.iterrows():
                if isinstance(row['quality'], dict):
                    score = row['quality'].get('overall', 0)
                    if score > 0:
                        quality_scores.append(score)
            
            if quality_scores:
                meta['quality_stats'] = {
                    'mean': round(sum(quality_scores) / len(quality_scores), 3),
                    'min': round(min(quality_scores), 3),
                    'max': round(max(quality_scores), 3)
                }
        
        meta_path = filepath.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
    
    def cleanup_old_files(self, days: int = 7):
        """Remove files older than specified days.
        
        Args:
            days: Number of days to keep files
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        removed_count = 0
        removed_size = 0
        
        for filepath in self.base_path.rglob("*.parquet"):
            if filepath.stat().st_mtime < cutoff_time:
                size = filepath.stat().st_size
                try:
                    filepath.unlink()
                    # Also remove metadata
                    meta_path = filepath.with_suffix('.meta.json')
                    if meta_path.exists():
                        meta_path.unlink()
                    
                    removed_count += 1
                    removed_size += size
                    logger.info(f"Removed old file: {filepath.name}")
                except Exception as e:
                    logger.error(f"Error removing {filepath}: {e}")
        
        if removed_count > 0:
            logger.info(
                f"Cleanup complete: removed {removed_count} files "
                f"({self._format_size(removed_size)})"
            )