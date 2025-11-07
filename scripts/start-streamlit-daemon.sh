#!/bin/bash

# Daemon-style Streamlit startup script
# This ensures Streamlit stays running even after SSH session ends

cd /home/ec2-user/Optimization

# Ensure PATH is set
export PATH="$HOME/.local/bin:$PATH"

# Kill any existing Streamlit processes
pkill -f "streamlit run" || true
sleep 2

# Clear log
> dashboard.log

# Start Streamlit using setsid to detach from terminal
# This ensures it runs even if the parent process exits
setsid bash -c "cd /home/ec2-user/Optimization && export PATH=\"\$HOME/.local/bin:\$PATH\" && python3 -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.runOnSave false --browser.serverAddress 0.0.0.0 --browser.gatherUsageStats false" > dashboard.log 2>&1 &

# Get the PID
STREAMLIT_PID=$!
echo $STREAMLIT_PID > streamlit.pid

# Wait a moment
sleep 3

# Check if process is still running
if ps -p $STREAMLIT_PID > /dev/null 2>&1; then
    echo "✅ Streamlit started with PID: $STREAMLIT_PID"
    echo "📝 Logs: tail -f dashboard.log"
    exit 0
else
    echo "❌ Streamlit process exited immediately"
    echo "📝 Checking logs:"
    cat dashboard.log
    exit 1
fi

