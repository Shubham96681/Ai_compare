#!/bin/bash
# Reliable Streamlit startup script for EC2
# This script ensures proper environment setup before starting Streamlit

set -e

PROJECT_DIR="/home/ec2-user/Optimization"
cd "$PROJECT_DIR"

# Ensure PATH includes user's local bin
export PATH="$HOME/.local/bin:$PATH"

# Ensure we're in the right directory
cd "$PROJECT_DIR"

# Start Streamlit
exec python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false

