# 🚀 CI/CD Deployment Status

## ✅ Current Status: **ACTIVE**

Your CI/CD pipeline is **fully configured** and ready to automatically deploy to EC2 whenever you push to the `main` branch.

---

## 📋 What's Configured

### GitHub Actions Workflow
- **File**: `.github/workflows/deploy.yml`
- **Trigger**: 
  - Automatic: Every push to `main` branch
  - Manual: Can be triggered from GitHub Actions tab
- **Jobs**:
  1. **Test Job**: Runs basic checks and validates imports
  2. **Deploy Job**: Deploys to EC2 after tests pass

### Deployment Process
1. ✅ Code is tested (imports, structure)
2. ✅ Code is deployed to EC2
3. ✅ Dependencies are installed/updated
4. ✅ Streamlit service is restarted
5. ✅ Health check verifies deployment

---

## 🔧 Required GitHub Secrets

Make sure these secrets are configured in your GitHub repository:

1. **EC2_HOST**
   - Your EC2 instance IP address
   - Example: `3.111.36.145`

2. **EC2_USER**
   - SSH username for EC2
   - Usually: `ec2-user` or `ubuntu`

3. **SSH_PRIVATE_KEY**
   - Your private SSH key content
   - Must include `-----BEGIN` and `-----END` lines

4. **EC2_SSH_PORT** (Optional)
   - SSH port (default: 22)

### How to Add Secrets:
1. Go to: https://github.com/infofitsoftwaresolution/Optimization/settings/secrets/actions
2. Click **New repository secret**
3. Add each secret with the name and value above

---

## 🚀 How to Deploy

### Automatic Deployment (Recommended)
Just push to main:
```bash
git add .
git commit -m "Your changes"
git push origin main
```

The workflow will automatically:
- Run tests
- Deploy to EC2
- Restart the application

### Manual Deployment
1. Go to: https://github.com/infofitsoftwaresolution/Optimization/actions
2. Select **CI/CD - Deploy to EC2**
3. Click **Run workflow**
4. Select branch: `main`
5. Click **Run workflow**

---

## 📊 Monitor Deployments

### GitHub Actions Dashboard
- **URL**: https://github.com/infofitsoftwaresolution/Optimization/actions
- **What to check**:
  - ✅ Green checkmark = Success
  - ❌ Red X = Failed (check logs)

### View Deployment Logs
1. Go to Actions tab
2. Click on the workflow run
3. Click on **Deploy to EC2** job
4. Expand steps to see detailed logs

---

## 🔍 Verify Deployment

### Check on EC2
```bash
# SSH into EC2
ssh ec2-user@YOUR_EC2_IP

# Check if Streamlit is running
ps aux | grep streamlit

# View logs
tail -f /home/ec2-user/Optimization/dashboard.log
# or
tail -f /home/ubuntu/Optimization/dashboard.log
```

### Check Application
- Open: `http://YOUR_EC2_IP:8501`
- Should see the Streamlit dashboard

---

## 🛠️ What the Workflow Does

### Test Job
- ✅ Checks out code
- ✅ Sets up Python 3.9
- ✅ Installs dependencies
- ✅ Validates imports (streamlit, pandas, boto3, numpy, plotly, openai)
- ✅ Checks Python version compatibility
- ✅ Validates project structure

### Deploy Job
- ✅ Connects to EC2 via SSH
- ✅ Finds project directory (supports both `/home/ec2-user` and `/home/ubuntu`)
- ✅ Backs up `.env` file
- ✅ Pulls latest code from GitHub
- ✅ Creates/updates virtual environment
- ✅ Installs/updates dependencies
- ✅ Clears Streamlit cache
- ✅ Restores `.env` file
- ✅ Restarts Streamlit (systemd or manual)
- ✅ Verifies deployment with health check
- ✅ Reports deployment status

---

## 🐛 Troubleshooting

### Deployment Fails

**Check GitHub Actions Logs:**
1. Go to Actions tab
2. Click on failed workflow
3. Check error messages in logs

**Common Issues:**

1. **SSH Connection Failed**
   - Verify `EC2_HOST` secret is correct
   - Verify `SSH_PRIVATE_KEY` secret is correct
   - Check EC2 security group allows SSH (port 22)

2. **Tests Failing**
   - Check import errors in test logs
   - Verify `requirements.txt` is up to date

3. **Streamlit Won't Start**
   - SSH into EC2 and check logs: `tail -50 dashboard.log`
   - Check if port 8501 is in use: `lsof -i :8501`
   - Verify Python dependencies: `pip list`

4. **Health Check Fails**
   - Check EC2 security group allows port 8501
   - Verify Streamlit is running: `ps aux | grep streamlit`
   - Check firewall: `sudo ufw status`

### Manual Fix on EC2

If deployment fails, you can manually fix on EC2:

```bash
# SSH into EC2
ssh ec2-user@YOUR_EC2_IP

# Navigate to project
cd /home/ec2-user/Optimization  # or /home/ubuntu/Optimization

# Pull latest code
git pull origin main

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Restart Streamlit
pkill -f "streamlit run"
nohup python3 -m streamlit run src/dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  > dashboard.log 2>&1 &
```

---

## 📝 Recent Improvements

### Latest Updates:
- ✅ Consolidated two workflows into one robust workflow
- ✅ Added support for both `ec2-user` and `ubuntu` users
- ✅ Improved error handling and logging
- ✅ Added automatic `.env` file backup/restore
- ✅ Added virtual environment support
- ✅ Added health check verification
- ✅ Added manual workflow dispatch option
- ✅ Improved deployment verification

---

## ✅ Deployment Checklist

Before deploying, ensure:
- [ ] GitHub secrets are configured (EC2_HOST, EC2_USER, SSH_PRIVATE_KEY)
- [ ] EC2 instance is running and accessible
- [ ] EC2 security group allows SSH (port 22) and HTTP (port 8501)
- [ ] Project is cloned on EC2 (or will be auto-created)
- [ ] `.env` file exists on EC2 (if needed)

---

## 🎯 Quick Reference

**Repository**: https://github.com/infofitsoftwaresolution/Optimization  
**Workflow File**: `.github/workflows/deploy.yml`  
**Trigger**: Push to `main` branch  
**Deployment Target**: EC2 instance  
**Dashboard URL**: `http://YOUR_EC2_IP:8501`

---

## 🎉 You're All Set!

Your CI/CD pipeline is ready. Just push to `main` and watch it deploy automatically!

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

