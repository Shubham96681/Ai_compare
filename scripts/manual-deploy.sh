#!/bin/bash

echo "🚀 Starting manual deployment..."

cd /home/ec2-user/Optimization

# Pull latest code
echo "📥 Pulling latest code..."
git pull origin main

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Kill existing process
echo "🛑 Stopping existing Streamlit app..."
pkill -f "streamlit run" || true
sleep 3

# Start new process
echo "🎯 Starting Streamlit app..."
export PATH="$HOME/.local/bin:$PATH"
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > dashboard.log 2>&1 &

echo "⏳ Waiting for app to start..."
sleep 8

# Check if running
if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Deployment completed!"
    echo "🌐 App is available at: http://3.111.36.145"
    echo "📝 Logs: tail -f /home/ec2-user/Optimization/dashboard.log"
else
    echo "❌ Deployment might have failed"
    echo "🔍 Check logs: tail -f /home/ec2-user/Optimization/dashboard.log"
fi
