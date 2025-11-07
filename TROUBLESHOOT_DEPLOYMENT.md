# 🔧 Troubleshooting Failed Deployment

## Quick Debug Steps

### Step 1: Check GitHub Actions Logs

In the GitHub Actions page, scroll down to see the error message. Look for:
- "❌ Failed to start Streamlit application"
- Any Python errors
- Permission errors

### Step 2: SSH into EC2 and Check Logs

```bash
ssh ec2-user@3.110.44.41
cd /home/ec2-user/Optimization
tail -50 dashboard.log
```

### Step 3: Check if Streamlit is Running

```bash
ps aux | grep streamlit
```

### Step 4: Try Starting Streamlit Manually

```bash
cd /home/ec2-user/Optimization
python3 -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

This will show any errors in real-time.

### Step 5: Check Python Path

```bash
which python3
python3 --version
which streamlit
```

### Common Issues:

1. **Streamlit not in PATH**: Use full path: `/home/ec2-user/.local/bin/streamlit`
2. **Port already in use**: Check with `netstat -tuln | grep 8501`
3. **Missing dependencies**: Check `pip3 list | grep streamlit`
4. **Permission errors**: Check file permissions

