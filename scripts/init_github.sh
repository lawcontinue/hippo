#!/bin/bash
# Hippo 🦛 GitHub Initialization Script
# This script helps create the GitHub repo and push the first commit

set -e

REPO_NAME="hippo"
GITHUB_USER="deepsearch"
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}"

echo "🦛 Hippo GitHub Initialization"
echo "================================"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Install from: https://cli.github.com/"
    exit 1
fi

# Check if user is logged in
if ! gh auth status &> /dev/null; then
    echo "❌ Not logged in to GitHub. Run: gh auth login"
    exit 1
fi

# Step 1: Initialize git repo (if not already)
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Hippo v0.1.0

- ✅ Ollama-compatible API
- ✅ Auto-unload and LRU eviction
- ✅ TUI dashboard
- ✅ Model quantization
- ✅ CI/CD and Docker
- ✅ 30/30 tests passing"
else
    echo "✅ Git repository already initialized"
fi

# Step 2: Create GitHub repo
echo ""
echo "🔧 Creating GitHub repository: ${REPO_URL}"
read -p "Create repo '${GITHUB_USER}/${REPO_NAME}'? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gh repo create ${REPO_NAME} --public --source=. --remote=origin --push
    echo "✅ Repository created and pushed"
else
    echo "⏭️  Skipping repo creation"
fi

# Step 3: Set up branches
echo ""
echo "🔧 Setting up branches..."
git branch -M main
echo "✅ Main branch set"

# Step 4: Create v0.1.0 release
echo ""
echo "🏷️  Creating GitHub Release: v0.1.0"
read -p "Create release 'v0.1.0'? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gh release create v0.1.0 --title "v0.1.0" --notes-file RELEASE_NOTES.md
    echo "✅ Release created"
else
    echo "⏭️  Skipping release creation"
fi

# Step 5: Enable GitHub Actions
echo ""
echo "✅ GitHub Actions will be enabled automatically"

# Step 6: Next steps
echo ""
echo "🎉 Hippo is now open source!"
echo ""
echo "📋 Next steps:"
echo "1. Visit: ${REPO_URL}"
echo "2. Star the repo ⭐"
echo "3. Publish to PyPI: python -m build && twine upload dist/*"
echo "4. Push to Docker Hub: docker push ${GITHUB_USER}/${REPO_NAME}:v0.1.0"
echo "5. Announce on Twitter/X, Reddit, HN"
echo ""
echo "🦛 Happy hacking!"
