"""
CLI module for TIDE 2.0.

This module provides the interactive Streamlit visualizer for inspecting clinical notes
before and after de-identification.

To launch the visualizer:
    tide2-visualizer [path]

The visualizer provides:
- Side-by-side inspection of original and de-identified notes with displacy entity highlights
- Tabular breakdown of detected PHI entities, character offsets, recognizers, and scores
- Filtering by entity type and recognizer engine
- Automatic detection of pipeline Parquet stage directories and Parquet files
"""

__all__ = []
