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
sudo systemctl reload nginx || true

# Stop existing Streamlit process gracefully
echo "🛑 Stopping existing Streamlit app..."
pkill -f "streamlit run" || true

# Wait for process to fully terminate
echo "⏳ Waiting for processes to terminate..."
sleep 10

# Force kill any remaining processes
pkill -9 -f "streamlit run" || true
sleep 2

# Clear any existing nohup output
> dashboard.log

# Start Streamlit app
echo "🎯 Starting Streamlit app..."
export PATH="$HOME/.local/bin:$PATH"
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.runOnSave false \
  --browser.serverAddress "0.0.0.0" \
  --browser.gatherUsageStats false \
  > dashboard.log 2>&1 &

# Get the PID
STREAMLIT_PID=$!
echo $STREAMLIT_PID > streamlit.pid

# Wait for app to start
echo "⏳ Waiting for app to start..."
for i in {1..30}; do
    if curl -f -s http://localhost:8501/ > /dev/null; then
        echo "✅ Streamlit app started successfully!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Streamlit app failed to start within 30 seconds"
        echo "📝 Check logs: tail -f dashboard.log"
        exit 1
    fi
    sleep 2
done

# Check if app is running
if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Deployment completed successfully!"
    echo "📊 App is running on: http://3.111.36.145"
    echo "📝 Check logs with: tail -f /home/ec2-user/Optimization/dashboard.log"
    echo "🔍 Process ID: $(cat streamlit.pid)"
else
    echo "❌ Deployment failed - Streamlit app not running"
    echo "🔍 Check logs: tail -f /home/ec2-user/Optimization/dashboard.log"
    exit 1
fi
