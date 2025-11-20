#!/usr/bin/env python3
"""
Study B Runner Script

Executes the complete Study B analysis and shows results.
"""

import subprocess
import sys
from pathlib import Path


def run_study_b():
    """Run complete Study B analysis"""

    script_dir = Path(__file__).parent

    print("🚀 Starting Study B Analysis...")
    print("=" * 60)

    # Run main analysis
    print("📊 Generating figures and tables...")
    result = subprocess.run(
        [sys.executable, script_dir / "study_b_plots.py"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("❌ Error in analysis:")
        print(result.stderr)
        return False

    print("✅ Analysis complete!")

    # Show summary
    print("\n📋 Analysis Summary:")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, script_dir / "study_b_summary.py"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print("⚠️  Warning: Could not generate summary")
        print(result.stderr)

    return True


if __name__ == "__main__":
    success = run_study_b()
    if success:
        print("\n🎉 Study B analysis completed successfully!")
        print(
            "📁 Check scripts/semantic_f1/study_b_outputs/ for all generated files"
        )
    else:
        print("\n❌ Study B analysis failed!")
        sys.exit(1)
