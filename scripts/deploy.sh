#!/bin/bash

# Deployment script for Streamlit app
set -e

echo "🚀 Starting deployment..."

# Navigate to app directory
cd /home/ec2-user/Optimization

# Pull latest changes
echo "📥 Pulling latest changes..."
git fetch origin
git reset --hard origin/main

# Install/update dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Restart nginx
echo "🔄 Reloading nginx..."
sudo systemctl reload nginx

# Kill existing Streamlit process
echo "🛑 Stopping existing Streamlit app..."
pkill -f "streamlit run" || true

# Wait a moment
sleep 5

# Start Streamlit app
echo "🎯 Starting Streamlit app..."
export PATH="$HOME/.local/bin:$PATH"
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > dashboard.log 2>&1 &

# Wait for app to start
sleep 8

# Check if app is running
if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Deployment completed successfully!"
    echo "📊 App is running on: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
    echo "📝 Check logs with: tail -f /home/ec2-user/Optimization/dashboard.log"
else
    echo "❌ Deployment failed - Streamlit app not running"
    echo "🔍 Check logs: tail -f /home/ec2-user/Optimization/dashboard.log"
    exit 1
fi
