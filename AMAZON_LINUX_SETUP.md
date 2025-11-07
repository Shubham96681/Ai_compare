# 🚀 Amazon Linux Setup - Quick Commands

You're on **Amazon Linux**, so use these commands:

---

## ✅ Run This on EC2 (Copy-Paste All)

```bash
# Update system
sudo yum update -y

# Install packages
sudo yum install -y python3 python3-pip git nginx curl

# Install Python dependencies
python3 -m pip install --upgrade pip setuptools wheel
pip3 install streamlit boto3 pandas numpy pydantic PyYAML plotly tiktoken requests tenacity python-dotenv sqlite-utils tqdm

# Setup project
mkdir -p /home/ec2-user/Optimization
cd /home/ec2-user/Optimization
git clone https://github.com/infofitsoftwaresolution/Optimization.git .
pip3 install -r requirements.txt

# Configure git
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"

# Setup nginx
sudo tee /etc/nginx/conf.d/optimization-app.conf > /dev/null <<'EOF'
server {
    listen 80;
    server_name 3.110.44.41;
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
EOF

# Test and start nginx
sudo nginx -t
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure firewall (if needed)
if command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-service=http
    sudo firewall-cmd --permanent --add-port=8501/tcp
    sudo firewall-cmd --reload
fi

echo "✅ Setup Complete!"
```

---

## ⚠️ Important: Update GitHub Secret

Make sure your GitHub secret `EC2_USER` is set to:
- **EC2_USER** = `ec2-user` (not `ubuntu`)

Go to: https://github.com/infofitsoftwaresolution/Optimization/settings/secrets/actions

---

## After Setup

1. **Exit EC2:** `exit`
2. **On your computer:** Commit and push
3. **Watch deployment:** GitHub Actions will deploy automatically

---

**That's it! Run the commands above on your EC2 instance.** 🚀

