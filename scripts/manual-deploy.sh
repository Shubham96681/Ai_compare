#!/bin/bash

echo "🚀 Starting manual deployment..."

cd /home/ec2-user/Optimization

# Pull latest code
git pull origin main

# Install dependencies
pip3 install -r requirements.txt

# Kill existing process
echo "🔄 Restarting Streamlit app..."
pkill -f "streamlit run" || true
sleep 3

# Start new process
export PATH="$HOME/.local/bin:$PATH"
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > dashboard.log 2>&1 &

echo "✅ Deployment completed!"
echo "📊 Check logs: tail -f dashboard.log"
echo "🌐 App should be available at: http://3.111.36.145:8501"

