#!/bin/bash

# Deployment script for Streamlit app
set -e

echo "Starting deployment..."

# Navigate to app directory
cd /home/ec2-user/Optimization

# Pull latest changes
echo "Pulling latest changes..."
git fetch origin
git reset --hard origin/main

# Install/update dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Restart services
echo "Restarting services..."
sudo systemctl reload nginx || true

# Kill existing Streamlit process
echo "Restarting Streamlit app..."
pkill -f "streamlit run" || true

# Wait a moment
sleep 2

# Start Streamlit app
export PATH="$HOME/.local/bin:$PATH"
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > dashboard.log 2>&1 &

echo "Deployment completed successfully!"
echo "Streamlit app is running on port 8501"

