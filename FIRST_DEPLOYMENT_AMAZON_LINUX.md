# 🚀 First Deployment - Amazon Linux Step-by-Step

You're on **Amazon Linux** (not Ubuntu), so use these commands instead!

---

## Part 1: Setup EC2 (One-Time)

### Step 1: You're Already Connected! ✅

You're on EC2 as `ec2-user`. Good!

### Step 2: Run Setup Commands

Copy and paste this **entire block** into your EC2 terminal:

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
if [ -f "nginx-optimization-app.conf" ]; then
    sudo cp nginx-optimization-app.conf /etc/nginx/conf.d/optimization-app.conf
    sudo rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true
else
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
fi

# Test and start nginx
sudo nginx -t
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure firewall (if firewalld is installed)
if command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --permanent --add-service=http
    sudo firewall-cmd --permanent --add-port=8501/tcp
    sudo firewall-cmd --reload
else
    echo "⚠️  Make sure EC2 security group allows ports 22, 80, and 8501"
fi

echo "✅ EC2 Setup Complete!"
```

**Wait for it to finish** (2-5 minutes).

### Step 3: Exit EC2

```bash
exit
```

---

## Part 2: Update GitHub Workflow for Amazon Linux

We need to update the deployment script to use the correct path for Amazon Linux.

### Step 4: Update Deployment Workflow

The workflow needs to use `/home/ec2-user/Optimization` instead of `/home/ubuntu/Optimization`.

I'll update the workflow file for you:

---

## Part 3: Deploy from Your Computer

### Step 5: Go to Your Project

```bash
cd D:\Optimization
```

### Step 6: Commit and Push

```bash
git add .
git commit -m "Setup: Complete CI/CD pipeline with GitHub Actions"
git push origin main
```

---

## Important Notes for Amazon Linux

1. **User:** `ec2-user` (not `ubuntu`)
2. **Package Manager:** `yum` (not `apt-get`)
3. **Project Path:** `/home/ec2-user/Optimization` (not `/home/ubuntu/Optimization`)
4. **Nginx Config:** `/etc/nginx/conf.d/` (not `/etc/nginx/sites-available/`)

---

## Next: Update GitHub Secrets

Make sure your GitHub secret `EC2_USER` is set to:
- `EC2_USER` = `ec2-user` (not `ubuntu`)

---

**Continue with the setup commands above!** 🚀

