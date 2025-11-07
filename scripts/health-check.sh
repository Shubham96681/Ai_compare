#!/bin/bash

# Health check for Streamlit app
echo "🔍 Performing health check..."

# Check if Streamlit process is running
if pgrep -f "streamlit run" > /dev/null; then
    echo "✅ Streamlit process is running"
    
    # Check if port 8501 is accessible
    if netstat -tuln | grep ':8501' > /dev/null; then
        echo "✅ Port 8501 is listening"
        
        # Try to curl the health endpoint (if available)
        if curl -f -s http://localhost:8501/ > /dev/null; then
            echo "✅ Streamlit app is responding"
            echo "🎉 Health check PASSED"
            exit 0
        else
            echo "⚠️  Streamlit app not responding on port 8501"
        fi
    else
        echo "❌ Port 8501 is not listening"
    fi
else
    echo "❌ Streamlit process not found"
fi

echo "💥 Health check FAILED"
exit 1
