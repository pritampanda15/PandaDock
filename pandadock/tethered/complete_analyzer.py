#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Analyzer for Tethered Docking

Provides end-to-end workflow combining ligand extraction, tethered docking,
pose validation, and comprehensive analysis for scoring function validation
and binding affinity prediction.

Author: Pritam Kumar Panda @ Stanford University
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from .ligand_extractor import LigandExtractor
from .tethered_docker import TetheredDocker
from .pose_validator import PoseValidator


class CompleteAnalyzer:
    """Complete end-to-end tethered docking analysis workflow"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extractor = LigandExtractor()
        self.docker = TetheredDocker()
        self.validator = PoseValidator()

    def run_complete_analysis(
        self,
        complex_file: str,
        ligand_id: str,
        chain_id: Optional[str] = None,
        tether_radius: float = 2.0,
        scoring_function: str = 'physics_based',
        algorithm: str = 'genetic',
        num_poses: int = 10,
        output_dir: str = '.',
        experimental_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete tethered docking analysis workflow

        Args:
            complex_file: PDB complex file with embedded ligand
            ligand_id: Ligand residue name to extract
            chain_id: Specific chain containing ligand (optional)
            tether_radius: Maximum distance from reference pose (Å)
            scoring_function: Scoring function for validation
            algorithm: Docking algorithm to use
            num_poses: Number of poses to generate
            output_dir: Output directory for all results
            experimental_data: Optional experimental binding data

        Returns:
            Comprehensive analysis results dictionary
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.logger.info("=" * 60)
            self.logger.info("PANDADOCK TETHERED ANALYSIS - COMPLETE WORKFLOW")
            self.logger.info("=" * 60)
            self.logger.info(f"Complex file: {complex_file}")
            self.logger.info(f"Target ligand: {ligand_id}")
            self.logger.info(f"Tether radius: {tether_radius} Å")
            self.logger.info(f"Scoring function: {scoring_function}")
            self.logger.info(f"Algorithm: {algorithm}")

            # Step 1: Extract ligand from complex
            self.logger.info("\n[1/5] LIGAND EXTRACTION")
            self.logger.info("-" * 30)

            extraction_dir = output_path / '01_extraction'
            extracted_files = self.extractor.extract_ligand_from_complex(
                complex_file=complex_file,
                ligand_id=ligand_id,
                chain_id=chain_id,
                output_dir=str(extraction_dir)
            )

            # Step 2: Validate crystal structure pose
            self.logger.info("\n[2/5] CRYSTAL POSE VALIDATION")
            self.logger.info("-" * 35)

            validation_dir = output_path / '02_validation'
            crystal_validation = self.validator.validate_poses(
                input_file=complex_file,
                ligand_id=ligand_id,
                scoring_function=scoring_function,
                reference_data=experimental_data,
                output_dir=str(validation_dir)
            )

            # Step 3: Run tethered docking
            self.logger.info("\n[3/5] TETHERED DOCKING")
            self.logger.info("-" * 25)

            docking_dir = output_path / '03_docking'
            docking_results = self.docker.run_tethered_docking(
                receptor_file=extracted_files['receptor'],
                ligand_file=extracted_files['ligand'],
                reference_pose=extracted_files['reference_pose'],
                tether_radius=tether_radius,
                scoring_function=scoring_function,
                algorithm=algorithm,
                output_dir=str(docking_dir),
                num_poses=num_poses
            )

            # Step 4: Validate docking results
            self.logger.info("\n[4/5] DOCKING RESULTS VALIDATION")
            self.logger.info("-" * 35)

            docking_validation_dir = output_path / '04_docking_validation'
            docking_validation = self.validator.validate_poses(
                input_file=str(docking_dir),
                scoring_function=scoring_function,
                reference_data=experimental_data,
                output_dir=str(docking_validation_dir)
            )

            # Step 5: Comprehensive analysis and comparison
            self.logger.info("\n[5/5] COMPREHENSIVE ANALYSIS")
            self.logger.info("-" * 32)

            analysis_results = self._perform_comprehensive_analysis(
                crystal_validation=crystal_validation,
                docking_results=docking_results,
                docking_validation=docking_validation,
                tether_radius=tether_radius,
                output_dir=output_path
            )

            # Generate final report
            final_results = self._generate_final_report(
                complex_file=complex_file,
                ligand_id=ligand_id,
                extraction_results=extracted_files,
                crystal_validation=crystal_validation,
                docking_results=docking_results,
                docking_validation=docking_validation,
                analysis_results=analysis_results,
                output_dir=output_path
            )

            self.logger.info("\n" + "=" * 60)
            self.logger.info("TETHERED ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)

            return final_results

        except Exception as e:
            self.logger.error(f"Complete analysis failed: {e}")
            raise

    def _perform_comprehensive_analysis(
        self,
        crystal_validation: Dict[str, Any],
        docking_results: Dict[str, Any],
        docking_validation: Dict[str, Any],
        tether_radius: float,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis comparing crystal and docking results"""

        analysis_dir = output_dir / '05_analysis'
        analysis_dir.mkdir(exist_ok=True)

        # Extract key metrics
        crystal_score = crystal_validation.get('crystal_score', 0)
        crystal_kd = crystal_validation.get('predicted_kd', float('inf'))

        best_docking_score = docking_results.get('best_score', 0)
        avg_rmsd = docking_results.get('avg_rmsd', 0)
        constraint_success_rate = docking_validation.get('constraint_success_rate', 0)

        # Score comparison analysis
        score_difference = abs(crystal_score - best_docking_score)
        score_agreement = self._assess_score_agreement(crystal_score, best_docking_score)

        # RMSD analysis
        rmsd_quality = self._assess_rmsd_quality(avg_rmsd, tether_radius)

        # Constraint satisfaction analysis
        constraint_quality = self._assess_constraint_quality(constraint_success_rate)

        # Overall validation assessment
        overall_assessment = self._assess_overall_validation(
            score_agreement, rmsd_quality, constraint_quality
        )

        analysis_results = {
            'score_comparison': {
                'crystal_score': crystal_score,
                'best_docking_score': best_docking_score,
                'score_difference': score_difference,
                'agreement_level': score_agreement
            },
            'pose_reproduction': {
                'average_rmsd': avg_rmsd,
                'tether_radius': tether_radius,
                'rmsd_quality': rmsd_quality
            },
            'constraint_satisfaction': {
                'success_rate': constraint_success_rate,
                'quality_assessment': constraint_quality
            },
            'binding_affinity_comparison': {
                'crystal_kd_predicted': crystal_kd,
                'crystal_kd_formatted': self._format_affinity(crystal_kd),
                'docking_kd_predicted': self._predict_kd_from_score(best_docking_score),
                'docking_kd_formatted': self._format_affinity(self._predict_kd_from_score(best_docking_score))
            },
            'validation_summary': {
                'overall_assessment': overall_assessment,
                'scoring_function_performance': self._assess_scoring_performance(score_agreement),
                'algorithm_performance': self._assess_algorithm_performance(rmsd_quality, constraint_quality)
            }
        }

        # Save detailed analysis
        analysis_file = analysis_dir / 'comprehensive_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)

        # Generate analysis plots and summaries
        self._generate_analysis_plots(analysis_results, analysis_dir)

        return analysis_results

    def _assess_score_agreement(self, crystal_score: float, docking_score: float) -> str:
        """Assess agreement between crystal and docking scores"""
        if abs(crystal_score) < 1e-6 or abs(docking_score) < 1e-6:
            return "Cannot assess (zero values)"

        relative_error = abs(crystal_score - docking_score) / abs(crystal_score)

        if relative_error < 0.1:
            return "Excellent"
        elif relative_error < 0.25:
            return "Good"
        elif relative_error < 0.5:
            return "Moderate"
        elif relative_error < 1.0:
            return "Poor"
        else:
            return "Very Poor"

    def _assess_rmsd_quality(self, avg_rmsd: float, tether_radius: float) -> str:
        """Assess RMSD quality relative to tether radius"""
        relative_rmsd = avg_rmsd / tether_radius if tether_radius > 0 else 1.0

        if relative_rmsd < 0.3:
            return "Excellent"
        elif relative_rmsd < 0.5:
            return "Good"
        elif relative_rmsd < 0.75:
            return "Moderate"
        elif relative_rmsd < 1.0:
            return "Poor"
        else:
            return "Very Poor"

    def _assess_constraint_quality(self, success_rate: float) -> str:
        """Assess constraint satisfaction quality"""
        if success_rate >= 0.9:
            return "Excellent"
        elif success_rate >= 0.75:
            return "Good"
        elif success_rate >= 0.5:
            return "Moderate"
        elif success_rate >= 0.25:
            return "Poor"
        else:
            return "Very Poor"

    def _assess_overall_validation(self, score_agreement: str, rmsd_quality: str, constraint_quality: str) -> str:
        """Assess overall validation quality"""
        quality_scores = {
            "Excellent": 5, "Good": 4, "Moderate": 3, "Poor": 2, "Very Poor": 1
        }

        scores = [
            quality_scores.get(score_agreement, 1),
            quality_scores.get(rmsd_quality, 1),
            quality_scores.get(constraint_quality, 1)
        ]

        avg_score = np.mean(scores)

        if avg_score >= 4.5:
            return "Excellent - Scoring function validates well"
        elif avg_score >= 3.5:
            return "Good - Scoring function performs adequately"
        elif avg_score >= 2.5:
            return "Moderate - Some issues with scoring function"
        elif avg_score >= 1.5:
            return "Poor - Significant scoring function problems"
        else:
            return "Very Poor - Scoring function requires major revision"

    def _assess_scoring_performance(self, score_agreement: str) -> str:
        """Assess scoring function performance"""
        if score_agreement in ["Excellent", "Good"]:
            return "Scoring function accurately reproduces crystal binding energy"
        elif score_agreement == "Moderate":
            return "Scoring function shows reasonable agreement with crystal data"
        else:
            return "Scoring function shows poor agreement - calibration needed"

    def _assess_algorithm_performance(self, rmsd_quality: str, constraint_quality: str) -> str:
        """Assess docking algorithm performance"""
        if rmsd_quality in ["Excellent", "Good"] and constraint_quality in ["Excellent", "Good"]:
            return "Algorithm effectively reproduces crystal pose within constraints"
        elif rmsd_quality in ["Excellent", "Good", "Moderate"]:
            return "Algorithm shows good pose reproduction capabilities"
        else:
            return "Algorithm struggles with pose reproduction - parameter tuning needed"

    def _predict_kd_from_score(self, score: float) -> float:
        """Predict Kd from docking score using thermodynamic relationship"""
        R = 0.001987  # kcal/mol/K
        T = 298.15    # K

        try:
            kd = np.exp(score / (R * T))
            return float(kd)
        except (OverflowError, ValueError):
            return float('inf')

    def _format_affinity(self, kd: float) -> str:
        """Format binding affinity for display"""
        if kd == float('inf') or kd > 1e10:
            return "No binding predicted"
        elif kd > 1e-3:
            return f"{kd:.2e} M (mM range)"
        elif kd > 1e-6:
            return f"{kd:.2e} M (μM range)"
        elif kd > 1e-9:
            return f"{kd:.2e} M (nM range)"
        elif kd > 1e-12:
            return f"{kd:.2e} M (pM range)"
        else:
            return f"{kd:.2e} M (very tight binding)"

    def _generate_analysis_plots(self, analysis_results: Dict[str, Any], output_dir: Path):
        """Generate analysis plots and visualizations"""

        # Create summary text files instead of actual plots for now
        summary_file = output_dir / 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("PANDADOCK TETHERED DOCKING - COMPREHENSIVE ANALYSIS\n")
            f.write("=" * 55 + "\n\n")

            # Score comparison section
            score_comp = analysis_results['score_comparison']
            f.write("SCORE COMPARISON\n")
            f.write("-" * 16 + "\n")
            f.write(f"Crystal structure score: {score_comp['crystal_score']:.3f} kcal/mol\n")
            f.write(f"Best docking score:      {score_comp['best_docking_score']:.3f} kcal/mol\n")
            f.write(f"Score difference:        {score_comp['score_difference']:.3f} kcal/mol\n")
            f.write(f"Agreement level:         {score_comp['agreement_level']}\n\n")

            # Pose reproduction section
            pose_repro = analysis_results['pose_reproduction']
            f.write("POSE REPRODUCTION\n")
            f.write("-" * 17 + "\n")
            f.write(f"Average RMSD:           {pose_repro['average_rmsd']:.3f} Å\n")
            f.write(f"Tether radius:          {pose_repro['tether_radius']:.1f} Å\n")
            f.write(f"RMSD quality:           {pose_repro['rmsd_quality']}\n\n")

            # Constraint satisfaction section
            constraint = analysis_results['constraint_satisfaction']
            f.write("CONSTRAINT SATISFACTION\n")
            f.write("-" * 23 + "\n")
            f.write(f"Success rate:           {constraint['success_rate']:.1%}\n")
            f.write(f"Quality assessment:     {constraint['quality_assessment']}\n\n")

            # Binding affinity section
            affinity = analysis_results['binding_affinity_comparison']
            f.write("BINDING AFFINITY COMPARISON\n")
            f.write("-" * 27 + "\n")
            f.write(f"Crystal Kd (predicted): {affinity['crystal_kd_formatted']}\n")
            f.write(f"Docking Kd (predicted): {affinity['docking_kd_formatted']}\n\n")

            # Overall assessment
            validation = analysis_results['validation_summary']
            f.write("VALIDATION SUMMARY\n")
            f.write("-" * 18 + "\n")
            f.write(f"Overall assessment:     {validation['overall_assessment']}\n")
            f.write(f"Scoring performance:    {validation['scoring_function_performance']}\n")
            f.write(f"Algorithm performance:  {validation['algorithm_performance']}\n")

        self.logger.info(f"Analysis summary saved: {summary_file}")

    def _generate_final_report(
        self,
        complex_file: str,
        ligand_id: str,
        extraction_results: Dict[str, str],
        crystal_validation: Dict[str, Any],
        docking_results: Dict[str, Any],
        docking_validation: Dict[str, Any],
        analysis_results: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate final comprehensive report"""

        final_report = {
            'tethered_docking_complete_analysis': {
                'input_parameters': {
                    'complex_file': str(complex_file),
                    'ligand_id': ligand_id,
                    'tether_radius': docking_results.get('tether_radius', 2.0),
                    'scoring_function': crystal_validation.get('scoring_function', 'unknown'),
                    'algorithm': 'genetic'  # Could be extracted from config
                },
                'workflow_results': {
                    'extraction': {
                        'success': True,
                        'extracted_files': extraction_results
                    },
                    'crystal_validation': crystal_validation,
                    'tethered_docking': docking_results,
                    'docking_validation': docking_validation,
                    'comprehensive_analysis': analysis_results
                },
                'key_findings': {
                    'score_reproduction': analysis_results['score_comparison']['agreement_level'],
                    'pose_reproduction': analysis_results['pose_reproduction']['rmsd_quality'],
                    'constraint_satisfaction': analysis_results['constraint_satisfaction']['quality_assessment'],
                    'overall_validation': analysis_results['validation_summary']['overall_assessment']
                },
                'output_directories': {
                    'extraction': str(output_dir / '01_extraction'),
                    'crystal_validation': str(output_dir / '02_validation'),
                    'tethered_docking': str(output_dir / '03_docking'),
                    'docking_validation': str(output_dir / '04_docking_validation'),
                    'comprehensive_analysis': str(output_dir / '05_analysis')
                }
            }
        }

        # Save final report
        report_file = output_dir / 'complete_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)

        self.logger.info(f"Final report saved: {report_file}")
        return final_report

    def quick_validation(
        self,
        complex_file: str,
        ligand_id: str,
        scoring_function: str = 'physics_based',
        output_dir: str = '.'
    ) -> Dict[str, Any]:
        """
        Quick validation workflow - just extract and validate crystal pose

        Args:
            complex_file: PDB complex file
            ligand_id: Ligand residue name
            scoring_function: Scoring function to use
            output_dir: Output directory

        Returns:
            Quick validation results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.logger.info("Running quick validation workflow")

            # Extract ligand
            extraction_dir = output_path / 'extraction'
            extracted_files = self.extractor.extract_ligand_from_complex(
                complex_file=complex_file,
                ligand_id=ligand_id,
                output_dir=str(extraction_dir)
            )

            # Validate crystal pose
            validation_results = self.validator.validate_poses(
                input_file=complex_file,
                ligand_id=ligand_id,
                scoring_function=scoring_function,
                output_dir=str(output_path)
            )

            # Combine results
            quick_results = {
                'extraction_successful': True,
                'extracted_files': extracted_files,
                'validation_results': validation_results,
                'crystal_score': validation_results.get('crystal_score'),
                'predicted_kd': validation_results.get('predicted_kd'),
                'predicted_ic50': validation_results.get('predicted_ic50'),
                'ligand_properties': validation_results.get('ligand_properties')
            }

            return quick_results

        except Exception as e:
            self.logger.error(f"Quick validation failed: {e}")
            raise