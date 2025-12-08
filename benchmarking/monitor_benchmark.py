#!/usr/bin/env python3
"""
Monitor comprehensive benchmarking progress in real-time

Usage:
    python monitor_benchmark.py benchmarking/comprehensive_benchmark_prepared_150
"""

import sys
import time
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

def monitor_benchmark(results_dir: Path, refresh_seconds: int = 30):
    """Monitor benchmark progress"""

    results_file = results_dir / "benchmark_results_intermediate.csv"

    print("="*80)
    print("PANDADOCK COMPREHENSIVE BENCHMARKING MONITOR")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Refresh interval: {refresh_seconds} seconds")
    print("Press Ctrl+C to stop monitoring (benchmark will continue)")
    print("="*80)
    print()

    start_time = time.time()
    last_count = 0

    while True:
        try:
            if not results_file.exists():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for results file to be created...")
                time.sleep(refresh_seconds)
                continue

            df = pd.read_csv(results_file)
            current_count = len(df)

            if current_count == last_count:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No new results yet... ({current_count}/1200)")
                time.sleep(refresh_seconds)
                continue

            # Clear screen and show progress
            print("\n" + "="*80)
            print(f"BENCHMARK PROGRESS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

            # Overall progress
            progress_pct = 100 * current_count / 1200
            print(f"\nOverall: {current_count}/1200 runs completed ({progress_pct:.1f}%)")

            # Progress bar
            bar_width = 50
            filled = int(bar_width * current_count / 1200)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"[{bar}]")

            # By algorithm
            print(f"\n{'Algorithm':<30} {'Completed':<12} {'Success Rate':<15} {'Avg Runtime'}")
            print("-" * 80)

            for algo in sorted(df['algorithm'].unique()):
                algo_df = df[df['algorithm'] == algo]
                total = len(algo_df)
                success = algo_df['success'].sum()
                success_rate = 100 * success / total if total > 0 else 0
                avg_runtime = algo_df[algo_df['success'] == True]['runtime_seconds'].mean() if success > 0 else 0

                print(f"{algo:<30} {total:<12} {success_rate:>5.1f}% ({success}/{total})    {avg_runtime:>6.1f}s")

            # Recent failures
            failures = df[df['success'] == False]
            if len(failures) > 0:
                print(f"\nRecent Failures: {len(failures)} total")
                error_counts = failures['error'].value_counts().head(3)
                for error, count in error_counts.items():
                    print(f"  - {error[:60]}...: {count}")

            # Time estimates
            elapsed = time.time() - start_time
            if current_count > last_count and elapsed > 0:
                rate = (current_count - last_count) / (elapsed / 60)  # runs per minute
                remaining_runs = 1200 - current_count
                if rate > 0:
                    eta_minutes = remaining_runs / rate
                    eta_time = datetime.now() + timedelta(minutes=eta_minutes)
                    print(f"\nRate: {rate:.2f} runs/minute")
                    print(f"Estimated completion: {eta_time.strftime('%Y-%m-%d %H:%M:%S')} ({eta_minutes/60:.1f} hours)")

            print("="*80)

            last_count = current_count
            time.sleep(refresh_seconds)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped (benchmark continues running)")
            print(f"Final count: {current_count}/1200 runs completed")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(refresh_seconds)

if __name__ == '__main__':
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarking/comprehensive_benchmark_prepared_150")
    monitor_benchmark(results_dir)
