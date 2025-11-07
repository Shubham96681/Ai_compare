# 🚀 First Deployment - Step by Step Guide

Follow these steps carefully for your first deployment.

---

## 📋 Prerequisites Checklist

Before starting, make sure you have:
- [x] GitHub secrets configured (EC2_HOST, EC2_USER, SSH_PRIVATE_KEY)
- [ ] SSH access to EC2 instance (3.110.44.41)
- [ ] EC2 instance is running
- [ ] EC2 security group allows:
  - Port 22 (SSH)
  - Port 8501 (Streamlit)

---

## Step 1: Verify EC2 Access

### 1.1 Test SSH Connection

Open your terminal/PowerShell and test SSH connection:

```bash
ssh ubuntu@3.110.44.41
```

**Expected result:** You should be able to connect to EC2.

**If connection fails:**
- Check EC2 instance is running
- Verify security group allows SSH (port 22) from your IP
- Check if you're using the correct SSH key

**If successful:** Type `exit` to return to your local machine.

---

## Step 2: Initial EC2 Setup

### 2.1 SSH into EC2

```bash
ssh ubuntu@3.110.44.41
```

### 2.2 Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2.3 Install Required System Packages

```bash
sudo apt-get install -y python3 python3-pip python3-venv git nginx curl
```

### 2.4 Install Python Dependencies

```bash
python3 -m pip install --upgrade pip setuptools wheel
pip3 install streamlit boto3 pandas numpy pydantic PyYAML plotly tiktoken requests tenacity python-dotenv sqlite-utils tqdm
```

### 2.5 Create Project Directory

```bash
mkdir -p /home/ubuntu/Optimization
cd /home/ubuntu/Optimization
```

### 2.6 Clone the Repository

```bash
git clone https://github.com/infofitsoftwaresolution/Optimization.git .
```

### 2.7 Install Project Dependencies

```bash
pip3 install -r requirements.txt
```

### 2.8 Configure Git (on EC2)

```bash
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"
```

### 2.9 Set Up Nginx

```bash
# Copy nginx config
sudo cp nginx-optimization-app.conf /etc/nginx/sites-available/optimization-app
sudo ln -sf /etc/nginx/sites-available/optimization-app /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Start nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### 2.10 Configure Firewall

```bash
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 8501/tcp # Streamlit
sudo ufw --force enable
```

### 2.11 Test Streamlit Manually (Optional)

```bash
cd /home/ubuntu/Optimization
python3 -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Press `Ctrl+C` to stop it. We'll let the deployment script handle this automatically.

### 2.12 Exit EC2

```bash
exit
```

---

## Step 3: Prepare Local Repository

### 3.1 Navigate to Your Project

```bash
cd D:\Optimization
```

### 3.2 Verify Git Configuration

```bash
git config user.email
git config user.name
```

**Should show:**
- Email: `infofitsoftware@gmail.com`
- Name: `InfoFit Software`

If not, set it:
```bash
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"
```

### 3.3 Check Current Status

```bash
git status
```

You should see all the CI/CD files we created.

---

## Step 4: Commit and Push CI/CD Setup

### 4.1 Stage All Files

```bash
git add .
```

### 4.2 Commit the Changes

```bash
git commit -m "Setup: Complete CI/CD pipeline with GitHub Actions and EC2 deployment"
```

### 4.3 Push to GitHub

```bash
git push origin main
```

**This will trigger the GitHub Actions workflow!**

---

## Step 5: Monitor Deployment

### 5.1 Open GitHub Actions

Go to: https://github.com/infofitsoftwaresolution/Optimization/actions

### 5.2 Watch the Workflow

1. You should see a new workflow run appear
2. Click on it to see details
3. Watch the progress:
   - **Test job** runs first (should complete quickly)
   - **Deploy job** runs after (this is the actual deployment)

### 5.3 Check Deployment Logs

Click on the "Deploy to EC2" step to see detailed logs.

**Look for:**
- ✅ "SSH connection successful"
- ✅ "Pulling latest code from main branch..."
- ✅ "Installing/upgrading Python dependencies..."
- ✅ "Streamlit application started successfully!"
- ✅ "Application is healthy and responding!"

**If you see errors:**
- Check the error message
- See troubleshooting section below

---

## Step 6: Verify Deployment

### 6.1 Check GitHub Actions Status

The workflow should show a green checkmark ✅ when complete.

### 6.2 Test Dashboard Access

Open your browser and go to:
```
http://3.110.44.41:8501
```

**Expected:** The Streamlit dashboard should load.

### 6.3 Verify on EC2 (Optional)

SSH into EC2 and check:

```bash
ssh ubuntu@3.110.44.41

# Check if Streamlit is running
ps aux | grep streamlit

# Check application logs
tail -f /home/ubuntu/Optimization/dashboard.log
```

Press `Ctrl+C` to exit the log view.

---

## Step 7: Test Automatic Updates

### 7.1 Make a Test Change

Edit any file, for example, add a comment to README.md:

```bash
# On your local machine
echo "" >> README.md
echo "<!-- Updated via CI/CD -->" >> README.md
```

### 7.2 Commit and Push

```bash
git add README.md
git commit -m "Test: Verify automatic deployment"
git push origin main
```

### 7.3 Watch It Deploy Again

1. Go to GitHub Actions
2. See the new workflow run
3. Wait for it to complete
4. Refresh the dashboard - your change should be there!

---

## 🎉 Success!

If everything worked:
- ✅ GitHub Actions deployed automatically
- ✅ Dashboard is accessible
- ✅ Future pushes will auto-deploy

---

## 🆘 Troubleshooting

### Problem: SSH Connection Failed in GitHub Actions

**Error:** "Permission denied (publickey)" or "Connection timeout"

**Solutions:**
1. Verify `SSH_PRIVATE_KEY` secret in GitHub:
   - Go to: Settings → Secrets → Actions
   - Check the key includes `-----BEGIN` and `-----END` lines
   - Make sure there are no extra spaces

2. Test SSH manually:
   ```bash
   ssh ubuntu@3.110.44.41
   ```

3. Check EC2 security group:
   - Allow SSH (port 22) from anywhere (0.0.0.0/0) or GitHub Actions IPs

### Problem: Deployment Fails - "Project directory not found"

**Error:** Directory doesn't exist on EC2

**Solution:** The workflow will create it automatically, but you can also:
```bash
ssh ubuntu@3.110.44.41
mkdir -p /home/ubuntu/Optimization
```

### Problem: Application Not Starting

**Error:** Streamlit fails to start

**Solutions:**
1. Check logs on EC2:
   ```bash
   ssh ubuntu@3.110.44.41
   tail -50 /home/ubuntu/Optimization/dashboard.log
   ```

2. Check if port 8501 is in use:
   ```bash
   sudo netstat -tuln | grep 8501
   ```

3. Check Python dependencies:
   ```bash
   pip3 list | grep streamlit
   ```

### Problem: Cannot Access Dashboard

**Error:** Browser can't connect to http://3.110.44.41:8501

**Solutions:**
1. Check EC2 security group:
   - Allow inbound traffic on port 8501 (or port 80 if using nginx)

2. Check if Streamlit is running:
   ```bash
   ssh ubuntu@3.110.44.41
   ps aux | grep streamlit
   ```

3. Check firewall:
   ```bash
   sudo ufw status
   ```

4. Test locally on EC2:
   ```bash
   curl http://localhost:8501
   ```

### Problem: Git Push Fails

**Error:** "Permission denied" or authentication error

**Solutions:**
1. Check git email is set:
   ```bash
   git config user.email
   ```

2. If using HTTPS, you may need to use a personal access token
3. If using SSH, check your SSH key is added to GitHub

---

## 📝 Quick Reference Commands

### On Your Local Machine:

```bash
# Check git config
git config user.email
git config user.name

# Set git config
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"

# Commit and push
git add .
git commit -m "Your message"
git push origin main
```

### On EC2:

```bash
# SSH into EC2
ssh ubuntu@3.110.44.41

# Check Streamlit status
ps aux | grep streamlit

# View logs
tail -f /home/ubuntu/Optimization/dashboard.log

# Restart manually (if needed)
cd /home/ubuntu/Optimization
pkill -f "streamlit run src/dashboard.py"
nohup python3 -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 > dashboard.log 2>&1 &
```

---

## ✅ Final Checklist

- [ ] EC2 instance is running
- [ ] SSH access works
- [ ] Initial EC2 setup completed
- [ ] Git configured locally
- [ ] CI/CD files committed and pushed
- [ ] GitHub Actions workflow ran successfully
- [ ] Dashboard is accessible
- [ ] Test deployment worked

---

**🎉 Congratulations! Your CI/CD pipeline is now set up and working!**

Every time you push to `main`, your application will automatically deploy to EC2.

