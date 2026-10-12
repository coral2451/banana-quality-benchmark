#!/usr/bin/env bash
# Reproduce every table and figure in the article and supplement.
# Usage: bash run_all.sh path/to/banana_quality.csv   (or .xlsx)
# Total run time is roughly 20 to 30 minutes on a laptop; the analysis step is the slowest.
set -e
python scripts/prepare_data.py "$1"
python scripts/banana_revision_analysis.py data/banana_quality.csv --out results   # Tables 3-10, S1-S3
python scripts/make_figures.py data/banana_quality.csv results                     # Figures 4, 5, 6
python scripts/make_supp_figures.py data/banana_quality.csv results                # Figure 3, S7-S12
python scripts/make_s5s6.py data/banana_quality.csv results                        # Figures S5, S6
python scripts/make_tree_fig.py data/banana_quality.csv results                    # Figure S13
echo "Done. Outputs are in results/"
