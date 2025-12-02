"""
PandaDock Analysis Module

Provides comprehensive analysis tools for molecular docking results including:
- Scoring function analysis and breakdowns
- Energy component calculations
- Binding affinity estimations
- Statistical analysis and reporting
"""

from .scoring_analysis import generate_comprehensive_scoring_report

__all__ = ['generate_comprehensive_scoring_report']