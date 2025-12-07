"""
src/__init__.py
Giúp export Pipeline ra ngoài package src
"""
# Cho phép import trực tiếp: from src import Pipeline
from .pipeline import Pipeline

__all__ = ['Pipeline']