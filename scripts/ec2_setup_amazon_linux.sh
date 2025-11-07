#!/bin/bash
# EC2 Server Setup Script for Optimization Project - Amazon Linux
# Run this script once on your EC2 instance to set up the environment

set -e

echo "🚀 Setting up EC2 server for Optimization project (Amazon Linux)..."
echo ""

# Update system packages
echo "📦 Step 1/8: Updating system packages..."
sudo yum update -y

# Install required system packages
echo ""
echo "📦 Step 2/8: Installing system dependencies..."
sudo yum install -y python3 python3-pip git nginx curl htop

# Install Python packages globally
echo ""
echo "📦 Step 3/8: Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
pip3 install streamlit boto3 pandas numpy pydantic PyYAML plotly tiktoken requests tenacity python-dotenv sqlite-utils tqdm

# Create project directory
PROJECT_DIR="/home/ec2-user/Optimization"
echo ""
echo "📁 Step 4/8: Setting up project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone repository if not already cloned
echo ""
echo "📥 Step 5/8: Cloning repository..."
if [ ! -d ".git" ]; then
    echo "Cloning repository..."
    git clone https://github.com/infofitsoftwaresolution/Optimization.git .
else
    echo "Repository already exists, pulling latest..."
    git pull origin main
fi

# Install project dependencies
echo ""
echo "📦 Step 6/8: Installing project dependencies..."
pip3 install -r requirements.txt

# Configure git
echo ""
echo "⚙️  Step 7/8: Configuring git..."
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"

# Set up nginx
echo ""
echo "🌐 Step 8/8: Configuring nginx and firewall..."

# Setup nginx
if [ -f "nginx-optimization-app.conf" ]; then
    sudo cp nginx-optimization-app.conf /etc/nginx/conf.d/optimization-app.conf
    # Remove default nginx config
    sudo rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true
else
    echo "⚠️  nginx-optimization-app.conf not found, creating basic config..."
    sudo tee /etc/nginx/conf.d/optimization-app.conf > /dev/null <<EOF
server {
    listen 80;
    server_name 3.110.44.41;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF
fi

# Test nginx configuration
sudo nginx -t

# Start and enable nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure firewall (Amazon Linux uses firewalld or security groups)
echo "🔥 Configuring firewall..."
# Check if firewalld is installed
if command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-service=http
    sudo firewall-cmd --permanent --add-service=https
    sudo firewall-cmd --permanent --add-port=8501/tcp
    sudo firewall-cmd --reload
else
    echo "⚠️  firewalld not found. Make sure EC2 security group allows ports 22, 80, and 8501"
fi

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

