#!/usr/bin/env bash
# run.sh -- run analysis, tests, and compile tex
# Usage (Git Bash / WSL / Linux / macOS):
#   ./run.sh

set -euo pipefail

# move to script directory (so relative paths work)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "Working directory: $HERE"

# 1) Run analysis pipeline (should produce images/ and images/... .tex table)
echo ">>> Running ProblemSet8_Kim.py ..."
python -u ProblemSet8_Kim.py

# 2) Compile LaTeX into PDF
TEXFILE="ProblemSet8_Kim.tex"
OUTPDF="${TEXFILE%.tex}.pdf"

if [ ! -f "$TEXFILE" ]; then
  echo "ERROR: $TEXFILE not found in $HERE" >&2
  exit 2
fi

echo ">>> Compiling LaTeX ($TEXFILE) -> PDF ..."
# run pdflatex twice to resolve references
pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"
pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"

echo ">>> Done. PDF output: $OUTPDF"
