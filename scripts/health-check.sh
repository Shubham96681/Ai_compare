#!/bin/bash

# Health check for Streamlit app
HEALTH_CHECK_URL="http://localhost:8501/_stcore/health"
MAX_RETRIES=30
RETRY_INTERVAL=10

echo "Performing health check..."

for i in $(seq 1 $MAX_RETRIES); do
    if curl -f -s "$HEALTH_CHECK_URL" > /dev/null; then
        echo "Health check passed! Application is running."
        exit 0
    fi
    echo "Health check attempt $i/$MAX_RETRIES failed. Retrying in $RETRY_INTERVAL seconds..."
    sleep $RETRY_INTERVAL
done

echo "Health check failed after $MAX_RETRIES attempts."
exit 1

