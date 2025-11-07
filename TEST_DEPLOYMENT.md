# 🧪 Test Deployment Guide

Since you've added the GitHub secrets, let's test the deployment!

## ✅ What You've Done

- ✅ Added `EC2_HOST` = `3.110.44.41`
- ✅ Added `EC2_USER` = `ubuntu`
- ✅ Added `SSH_PRIVATE_KEY` = Your EC2 SSH private key

## 🚀 Test the Deployment

### Option 1: Quick Test (Recommended)

1. **Make a small test change:**
   ```bash
   # Add a comment or update a file
   echo "# Test deployment" >> README.md
   ```

2. **Commit and push:**
   ```bash
   git add .
   git commit -m "Test: Verify CI/CD deployment"
   git push origin main
   ```

3. **Monitor the deployment:**
   - Go to: https://github.com/infofitsoftwaresolution/Optimization/actions
   - Click on the latest workflow run
   - Watch the deployment progress

4. **Verify the dashboard:**
   - After deployment completes, check: http://3.110.44.41:8501
   - The dashboard should be accessible

### Option 2: Manual Test (If EC2 Setup Not Done)

If you haven't run the EC2 setup script yet:

1. **SSH into EC2:**
   ```bash
   ssh ubuntu@3.110.44.41
   ```

2. **Run the setup script:**
   ```bash
   cd ~
   # Copy the ec2_setup.sh script or clone the repo
   git clone https://github.com/infofitsoftwaresolution/Optimization.git
   cd Optimization
   chmod +x scripts/ec2_setup.sh
   ./scripts/ec2_setup.sh
   ```

3. **Then follow Option 1 above**

## 🔍 Verify Deployment Status

### Check GitHub Actions

1. Go to: https://github.com/infofitsoftwaresolution/Optimization/actions
2. Look for the latest workflow run
3. Check if it shows:
   - ✅ Green checkmark = Success
   - ❌ Red X = Failed (check logs)

### Check EC2 Manually

```bash
# SSH into EC2
ssh ubuntu@3.110.44.41

# Check if Streamlit is running
ps aux | grep streamlit

# Check application logs
tail -f /home/ubuntu/Optimization/dashboard.log

# Check if port 8501 is listening
netstat -tuln | grep 8501
```

## 🐛 Troubleshooting

### ❌ Deployment Fails: "Permission denied (publickey)"

**Problem:** SSH key authentication failed

**Solution:**
1. Verify the SSH private key in GitHub secrets is correct
2. Make sure it includes the full key (BEGIN and END lines)
3. Test SSH manually: `ssh ubuntu@3.110.44.41`

### ❌ Deployment Fails: "Connection timeout"

**Problem:** Cannot connect to EC2

**Solution:**
1. Check EC2 instance is running
2. Verify security group allows SSH (port 22) from GitHub Actions IPs
3. Check EC2 public IP is correct: `3.110.44.41`

### ❌ Application Not Starting

**Problem:** Deployment succeeds but app doesn't start

**Solution:**
1. SSH into EC2 and check logs: `tail -f /home/ubuntu/Optimization/dashboard.log`
2. Check if Python dependencies are installed
3. Verify port 8501 is not in use by another process

### ❌ Cannot Access Dashboard

**Problem:** Dashboard not accessible at http://3.110.44.41:8501

**Solution:**
1. Check EC2 security group allows inbound traffic on port 8501
2. Verify Streamlit is running: `ps aux | grep streamlit`
3. Check firewall: `sudo ufw status`
4. Test locally on EC2: `curl http://localhost:8501`

## 📊 Expected Workflow Output

When deployment succeeds, you should see:

```
🚀 Starting automated deployment...
📅 Deployment time: [timestamp]
🔗 Commit: [commit hash]
👤 Committed by: [your username]
📥 Pulling latest code from main branch...
📦 Installing/upgrading Python dependencies...
🧹 Clearing Streamlit cache...
🔄 Restarting application...
✅ Streamlit application started successfully!
📊 Application status:
🌐 Checking if application is responding...
✅ Application is healthy and responding!
✅ Deployment completed successfully!
```

## ✅ Success Checklist

- [ ] GitHub Actions workflow runs without errors
- [ ] Deployment job completes successfully
- [ ] Dashboard is accessible at http://3.110.44.41:8501
- [ ] Application logs show no errors
- [ ] Changes are reflected on the deployed version

## 🎉 Next Steps After Successful Test

Once deployment works:

1. **Make your actual changes**
2. **Commit and push** - deployment happens automatically!
3. **Monitor** via GitHub Actions
4. **Access** your updated dashboard

---

**Ready to test? Make a small change and push to main!** 🚀

