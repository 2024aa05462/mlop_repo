#!/bin/bash
# Pre-commit hook for Heart Disease MLOps Project
# This script runs flake8 linting before each commit
#
# To install this hook, run:
#   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or use pre-commit framework:
#   pip install pre-commit
#   pre-commit install

set -e

echo "🔍 Running pre-commit checks..."

# Check if flake8 is installed
if ! command -v flake8 &> /dev/null && ! python3 -m flake8 --version &> /dev/null; then
    echo "❌ flake8 is not installed. Install it with: pip install flake8"
    exit 1
fi

# Run flake8 for critical errors (blocking)
echo "📋 Checking for critical errors (E9, F63, F7, F82)..."
python3 -m flake8 src/ api/ tests/ \
    --count \
    --select=E9,F63,F7,F82 \
    --show-source \
    --statistics

if [ $? -ne 0 ]; then
    echo "❌ Critical flake8 errors found. Please fix them before committing."
    exit 1
fi

echo "✅ No critical errors found."

# Run flake8 for style warnings (non-blocking)
echo "📋 Checking for style warnings..."
python3 -m flake8 src/ api/ tests/ \
    --count \
    --exit-zero \
    --max-complexity=10 \
    --max-line-length=120 \
    --statistics

echo "✅ Pre-commit checks passed!"
exit 0

