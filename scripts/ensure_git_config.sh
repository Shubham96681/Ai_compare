#!/bin/bash
# Ensure Git Configuration Script
# Run this before committing to ensure correct git user email

CURRENT_EMAIL=$(git config user.email)
DESIRED_EMAIL="infofitsoftware@gmail.com"

if [ "$CURRENT_EMAIL" != "$DESIRED_EMAIL" ]; then
    echo "⚠️  Git email is not set correctly. Current: $CURRENT_EMAIL"
    echo "🔧 Setting git email to: $DESIRED_EMAIL"
    git config user.email "$DESIRED_EMAIL"
    git config user.name "InfoFit Software"
    echo "✅ Git configuration updated!"
else
    echo "✅ Git email is correctly set: $CURRENT_EMAIL"
fi

