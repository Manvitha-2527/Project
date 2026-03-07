#!/bin/bash

# Run this script from your Project Root directory
# Usage: bash git_push.sh "your commit message"

COMMIT_MSG=${1:-"Add project files"}

echo "📁 Initializing git (if not already)..."
git init

echo "📦 Staging all files..."
git add \
  Data/ \
  Visualizations/ \
  Dataset/ \
  models/ \
  scripts/ \
  README.md \
  requirements.txt \
  report.pdf

echo "💬 Committing with message: '$COMMIT_MSG'"
git commit -m "$COMMIT_MSG"

echo "🚀 Pushing to remote..."
git push origin main

echo "✅ Done!"
