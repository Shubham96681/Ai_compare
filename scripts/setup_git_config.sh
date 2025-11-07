#!/bin/bash
# Git Configuration Script
# Sets up git user email for commits

echo "⚙️  Configuring git user settings..."

# Set global git config for this repository
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"

# Verify configuration
echo ""
echo "✅ Git configuration updated:"
echo "   Email: $(git config user.email)"
echo "   Name: $(git config user.name)"
echo ""
echo "💡 Note: This configuration is set for this repository only."
echo "   For global configuration, use: git config --global user.email 'infofitsoftware@gmail.com'"

