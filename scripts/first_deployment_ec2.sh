#!/bin/bash
# First Deployment EC2 Setup Script
# Copy and paste this entire script into your EC2 terminal

set -e

echo "🚀 Starting First Deployment Setup on EC2..."
echo ""

# Step 1: Update system
echo "📦 Step 1/8: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install system packages
echo ""
echo "📦 Step 2/8: Installing system dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git nginx curl ufw

# Step 3: Install Python dependencies
echo ""
echo "📦 Step 3/8: Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
pip3 install streamlit boto3 pandas numpy pydantic PyYAML plotly tiktoken requests tenacity python-dotenv sqlite-utils tqdm

# Step 4: Create project directory
echo ""
echo "📁 Step 4/8: Setting up project directory..."
mkdir -p /home/ubuntu/Optimization
cd /home/ubuntu/Optimization

# Step 5: Clone repository
echo ""
echo "📥 Step 5/8: Cloning repository..."
if [ -d ".git" ]; then
    echo "Repository already exists, pulling latest..."
    git pull origin main
else
    git clone https://github.com/infofitsoftwaresolution/Optimization.git .
fi

# Step 6: Install project dependencies
echo ""
echo "📦 Step 6/8: Installing project dependencies..."
pip3 install -r requirements.txt

# Step 7: Configure git
echo ""
echo "⚙️  Step 7/8: Configuring git..."
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"

# Step 8: Setup nginx and firewall
echo ""
echo "🌐 Step 8/8: Configuring nginx and firewall..."

# Setup nginx
sudo cp nginx-optimization-app.conf /etc/nginx/sites-available/optimization-app 2>/dev/null || echo "Nginx config file not found, skipping..."
sudo ln -sf /etc/nginx/sites-available/optimization-app /etc/nginx/sites-enabled/ 2>/dev/null || true
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
sudo nginx -t

# Start nginx
sudo systemctl restart nginx
sudo systemctl enable nginx

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 8501/tcp
sudo ufw --force enable

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ EC2 Setup Complete!"
echo ""
echo "📋 Next steps:"
echo "1. Exit EC2: type 'exit'"
echo "2. On your local machine, commit and push:"
echo "   git add ."
echo "   git commit -m 'Setup: CI/CD pipeline'"
echo "   git push origin main"
echo "3. Watch deployment at:"
echo "   https://github.com/infofitsoftwaresolution/Optimization/actions"
echo "4. Access dashboard at:"
echo "   http://3.110.44.41:8501"
echo ""

