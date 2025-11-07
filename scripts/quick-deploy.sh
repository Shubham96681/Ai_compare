#!/bin/bash

# Simple, reliable deployment script
set -e

cd /home/ec2-user/Optimization

# Pull latest
git pull origin main

# Install deps
pip3 install -r requirements.txt --quiet

# Stop old process
pkill -f "streamlit run" || true
sleep 3
pkill -9 -f "streamlit run" || true
sleep 2

# Start new process - simple and direct
export PATH="$HOME/.local/bin:$PATH"
cd /home/ec2-user/Optimization
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.runOnSave false \
  --browser.serverAddress 0.0.0.0 \
  --browser.gatherUsageStats false \
  > dashboard.log 2>&1 &

# Wait and verify
sleep 8

if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Streamlit is running"
    exit 0
else
    echo "❌ Streamlit failed to start"
    tail -30 dashboard.log
    exit 1
fi

