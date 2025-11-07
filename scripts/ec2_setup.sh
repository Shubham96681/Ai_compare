#!/bin/bash
# EC2 Server Setup Script for Optimization Project
# Run this script once on your EC2 instance to set up the environment

set -e

echo "🚀 Setting up EC2 server for Optimization project..."
echo ""

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required system packages
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    nginx \
    curl \
    htop \
    ufw

# Install Python packages globally (or use venv)
echo "📦 Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel
pip3 install streamlit boto3 pandas numpy pydantic PyYAML plotly tiktoken requests tenacity python-dotenv sqlite-utils tqdm

# Create project directory
PROJECT_DIR="/home/ubuntu/Optimization"
echo "📁 Setting up project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone repository if not already cloned
if [ ! -d ".git" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/infofitsoftwaresolution/Optimization.git .
else
    echo "📥 Repository already exists, pulling latest..."
    git pull origin main
fi

# Configure git (if needed)
echo "⚙️  Configuring git..."
git config user.email "infofitsoftware@gmail.com" || true
git config user.name "InfoFit Software" || true

# Set up nginx
echo "🌐 Configuring nginx..."
sudo cp nginx-optimization-app.conf /etc/nginx/sites-available/optimization-app
sudo ln -sf /etc/nginx/sites-available/optimization-app /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default  # Remove default site

# Test nginx configuration
sudo nginx -t

# Configure firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS (if using SSL later)
sudo ufw --force enable

# Create systemd service for Streamlit (optional, better than nohup)
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/optimization-app.service > /dev/null <<EOF
[Unit]
Description=Optimization Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable optimization-app.service

# Start services
echo "🔄 Starting services..."
sudo systemctl restart nginx
sudo systemctl start optimization-app || echo "⚠️  Service start failed, will use manual start"

echo ""
echo "✅ EC2 setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Configure GitHub Secrets (see DEPLOYMENT.md)"
echo "2. Test deployment by pushing to main branch"
echo "3. Access dashboard at: http://$(curl -s ifconfig.me):8501"
echo ""
echo "🔍 Useful commands:"
echo "  - Check Streamlit status: sudo systemctl status optimization-app"
echo "  - View Streamlit logs: sudo journalctl -u optimization-app -f"
echo "  - Restart Streamlit: sudo systemctl restart optimization-app"
echo "  - Check nginx status: sudo systemctl status nginx"
echo "  - View nginx logs: sudo tail -f /var/log/nginx/error.log"

