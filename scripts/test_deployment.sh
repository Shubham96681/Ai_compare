#!/bin/bash
# Test Deployment Script
# This script tests the SSH connection and deployment setup on EC2

set -e

echo "🧪 Testing EC2 Deployment Setup..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test SSH connection
echo "1️⃣  Testing SSH connection..."
if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@3.110.44.41 "echo 'SSH connection successful'" 2>/dev/null; then
    echo -e "${GREEN}✅ SSH connection successful!${NC}"
else
    echo -e "${RED}❌ SSH connection failed!${NC}"
    echo "   Please check:"
    echo "   - EC2 instance is running"
    echo "   - Security group allows SSH (port 22)"
    echo "   - SSH key is correct"
    exit 1
fi

# Test project directory
echo ""
echo "2️⃣  Checking project directory..."
if ssh ubuntu@3.110.44.41 "test -d /home/ubuntu/Optimization" 2>/dev/null; then
    echo -e "${GREEN}✅ Project directory exists${NC}"
else
    echo -e "${YELLOW}⚠️  Project directory not found${NC}"
    echo "   The deployment script will create it automatically"
fi

# Test Python installation
echo ""
echo "3️⃣  Checking Python installation..."
PYTHON_VERSION=$(ssh ubuntu@3.110.44.41 "python3 --version 2>&1" 2>/dev/null || echo "not found")
if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
    echo -e "${GREEN}✅ Python installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python not found${NC}"
    echo "   Run the ec2_setup.sh script first"
fi

# Test Git installation
echo ""
echo "4️⃣  Checking Git installation..."
GIT_VERSION=$(ssh ubuntu@3.110.44.41 "git --version 2>&1" 2>/dev/null || echo "not found")
if [[ $GIT_VERSION == *"git version"* ]]; then
    echo -e "${GREEN}✅ Git installed: $GIT_VERSION${NC}"
else
    echo -e "${RED}❌ Git not found${NC}"
    echo "   Run the ec2_setup.sh script first"
fi

# Test if Streamlit is running
echo ""
echo "5️⃣  Checking if Streamlit is running..."
if ssh ubuntu@3.110.44.41 "pgrep -f 'streamlit run src/dashboard.py' > /dev/null" 2>/dev/null; then
    echo -e "${GREEN}✅ Streamlit is running${NC}"
else
    echo -e "${YELLOW}⚠️  Streamlit is not running${NC}"
    echo "   This is normal if you haven't deployed yet"
fi

# Test port 8501
echo ""
echo "6️⃣  Checking port 8501..."
if ssh ubuntu@3.110.44.41 "netstat -tuln | grep :8501 > /dev/null" 2>/dev/null; then
    echo -e "${GREEN}✅ Port 8501 is in use${NC}"
else
    echo -e "${YELLOW}⚠️  Port 8501 is not in use${NC}"
    echo "   This is normal if Streamlit is not running"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Deployment setup test completed!${NC}"
echo ""
echo "📋 Next steps:"
echo "   1. Push code to main branch to trigger deployment"
echo "   2. Monitor deployment at: https://github.com/infofitsoftwaresolution/Optimization/actions"
echo "   3. Access dashboard at: http://3.110.44.41:8501"
echo ""

