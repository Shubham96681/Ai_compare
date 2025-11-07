# 🚀 First Deployment - Simple Step-by-Step

## Quick Overview

1. **Setup EC2** (one-time)
2. **Commit & Push** (triggers deployment)
3. **Done!** ✅

---

## Part 1: Setup EC2 (One-Time)

### Step 1: Connect to EC2

Open your terminal/PowerShell and run:

```bash
ssh ubuntu@3.110.44.41
```

**If it asks for a password or key:** Use your EC2 SSH key.

### Step 2: Run Setup Script

Once connected to EC2, copy and paste this entire block:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install packages
sudo apt-get install -y python3 python3-pip python3-venv git nginx curl ufw

# Install Python dependencies
python3 -m pip install --upgrade pip setuptools wheel
pip3 install streamlit boto3 pandas numpy pydantic PyYAML plotly tiktoken requests tenacity python-dotenv sqlite-utils tqdm

# Setup project
mkdir -p /home/ubuntu/Optimization
cd /home/ubuntu/Optimization
git clone https://github.com/infofitsoftwaresolution/Optimization.git .
pip3 install -r requirements.txt

# Configure git
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"

# Setup nginx
sudo cp nginx-optimization-app.conf /etc/nginx/sites-available/optimization-app
sudo ln -sf /etc/nginx/sites-available/optimization-app /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# Setup firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 8501/tcp
sudo ufw --force enable

echo "✅ EC2 Setup Complete!"
```

**Wait for it to finish** (may take 2-5 minutes).

### Step 3: Exit EC2

```bash
exit
```

---

## Part 2: Deploy from Your Computer

### Step 4: Go to Your Project Folder

```bash
cd D:\Optimization
```

### Step 5: Check Git Config

```bash
git config user.email
```

**Should show:** `infofitsoftware@gmail.com`

If not, set it:
```bash
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"
```

### Step 6: Commit All Changes

```bash
git add .
git commit -m "Setup: Complete CI/CD pipeline with GitHub Actions"
```

### Step 7: Push to GitHub

```bash
git push origin main
```

**This triggers the deployment!** 🚀

---

## Part 3: Watch It Deploy

### Step 8: Open GitHub Actions

Go to: **https://github.com/infofitsoftwaresolution/Optimization/actions**

### Step 9: Watch the Workflow

1. You'll see a new workflow run
2. Click on it
3. Watch it deploy:
   - ✅ Test job (quick)
   - ✅ Deploy job (pulls code, installs, starts app)

**Wait 2-3 minutes** for it to complete.

### Step 10: Check Your Dashboard

Open in browser:
```
http://3.110.44.41:8501
```

**You should see your Streamlit dashboard!** 🎉

---

## ✅ That's It!

From now on, every time you:
```bash
git push origin main
```

Your app will automatically deploy to EC2!

---

## 🆘 Troubleshooting

### Can't SSH to EC2?
- Check EC2 is running
- Check security group allows SSH (port 22)
- Verify IP: `3.110.44.41`

### Deployment Failed?
- Check GitHub Actions logs
- Verify GitHub secrets are set correctly
- See [FIRST_DEPLOYMENT_STEPS.md](FIRST_DEPLOYMENT_STEPS.md) for detailed troubleshooting

### Dashboard Not Loading?
- Check EC2 security group allows port 8501
- Wait 1-2 minutes after deployment
- Try: `http://3.110.44.41:8501`

---

**Need more details?** See [FIRST_DEPLOYMENT_STEPS.md](FIRST_DEPLOYMENT_STEPS.md)

