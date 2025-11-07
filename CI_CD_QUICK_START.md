# 🚀 CI/CD Quick Start Guide

## ✅ What's Been Set Up

1. **GitHub Actions Workflow** (`.github/workflows/deploy.yml`)
   - Automatically deploys on push to `main` branch
   - Runs tests before deployment
   - Deploys to EC2 instance at `3.110.44.41`

2. **Git Configuration**
   - Email: `infofitsoftware@gmail.com`
   - Name: `InfoFit Software`

3. **EC2 Deployment Scripts**
   - `scripts/ec2_setup.sh` - Initial EC2 server setup
   - `scripts/setup_git_config.sh` - Git configuration helper

## 🎯 Quick Commands

### Before Pushing Code:

**Windows PowerShell:**
```powershell
# Ensure git email is set correctly
.\scripts\ensure_git_config.ps1

# Or manually:
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"
```

**Linux/Mac:**
```bash
# Ensure git email is set correctly
chmod +x scripts/ensure_git_config.sh
./scripts/ensure_git_config.sh

# Or manually:
git config user.email "infofitsoftware@gmail.com"
git config user.name "InfoFit Software"
```

### Push to Deploy:

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

**That's it!** The deployment will happen automatically via GitHub Actions.

## 🔐 Required GitHub Secrets

Make sure these are set in GitHub → Settings → Secrets → Actions:

- `EC2_HOST` = `3.110.44.41`
- `EC2_USER` = `ubuntu`
- `SSH_PRIVATE_KEY` = Your EC2 SSH private key

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup instructions.

## 📍 Important URLs

- **GitHub Repo**: https://github.com/infofitsoftwaresolution/Optimization
- **EC2 Dashboard**: http://3.110.44.41:8501
- **GitHub Actions**: https://github.com/infofitsoftwaresolution/Optimization/actions

## 🔍 Check Deployment Status

1. Go to: https://github.com/infofitsoftwaresolution/Optimization/actions
2. Click on the latest workflow run
3. View deployment logs

## 🆘 Troubleshooting

**Git email not set?**
```bash
git config user.email "infofitsoftware@gmail.com"
```

**Deployment failed?**
- Check GitHub Actions logs
- Verify GitHub secrets are set correctly
- SSH into EC2 and check logs: `tail -f /home/ubuntu/Optimization/dashboard.log`

**Can't access dashboard?**
- Check EC2 security group allows port 8501
- Verify Streamlit is running: `ps aux | grep streamlit`

For more details, see [DEPLOYMENT.md](DEPLOYMENT.md).

