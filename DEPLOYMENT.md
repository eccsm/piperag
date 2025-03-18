# Piperag API Server Deployment Guide

This guide provides instructions for deploying the Piperag API server to be used with the CC Resume App.

## Prerequisites

- Ubuntu 20.04 LTS or newer VPS/VM with at least 4GB RAM and 20GB storage
- A domain name pointing to your server (e.g., `api.yourdomain.com`)
- Basic familiarity with Linux commands

## Setup Steps

### 1. Initial Server Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git nginx fail2ban ufw

# Configure firewall
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Enable fail2ban for SSH protection
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. Clone the Repository

```bash
# Create a dedicated user (optional but recommended)
sudo adduser piperag_admin
sudo usermod -aG sudo piperag_admin
su - piperag_admin

# Clone the repository
git clone https://github.com/yourusername/piperag.git
cd piperag

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn uvicorn
```

### 3. Configure Environment Variables

Create a `.env` file with your configuration:

```bash
nano .env
```

Add the following content, replacing placeholder values:

```
# API Security
API_KEY=your-long-randomly-generated-api-key-here

# Model configurations
MODEL_TYPE=vicuna_ggml
GGUF_CACHE_DIR=/home/piperag_admin/piperag/vicuna-7b-q8
CHROMA_DIR=/home/piperag_admin/piperag/chroma_db
```

Generate a secure API key:

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

### 4. Download Model Files

Set up directories and download model files:

```bash
mkdir -p vicuna-7b-q8
mkdir -p chroma_db

# Download model files (example - adjust based on your model source)
# This step depends on where your model files are stored
# You might need to download from Hugging Face or another source
```

### 5. Configure Nginx as Reverse Proxy

Create an Nginx configuration file:

```bash
sudo nano /etc/nginx/sites-available/piperag
```

Add the following configuration, replacing `api.yourdomain.com` with your domain:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Add security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    }
    
    # Deny access to dot files
    location ~ /\. {
        deny all;
    }
}
```

Enable the site and test Nginx configuration:

```bash
sudo ln -s /etc/nginx/sites-available/piperag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Set Up SSL with Certbot

Install Certbot and obtain SSL certificate:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

Follow the prompts to complete the SSL setup.

### 7. Create Systemd Service

Create a service file to manage the application:

```bash
sudo nano /etc/systemd/system/piperag.service
```

Add the following content:

```ini
[Unit]
Description=Piperag API Service
After=network.target

[Service]
User=piperag_admin
Group=piperag_admin
WorkingDirectory=/home/piperag_admin/piperag
ExecStart=/home/piperag_admin/piperag/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 127.0.0.1:8000
Restart=always
# Security measures
PrivateTmp=true
ProtectSystem=full
NoNewPrivileges=true
# Environment
EnvironmentFile=/home/piperag_admin/piperag/.env

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable piperag
sudo systemctl start piperag
sudo systemctl status piperag
```

### 8. Monitoring and Log Setup

Set up log directory and permissions:

```bash
mkdir -p /home/piperag_admin/piperag/logs
touch /home/piperag_admin/piperag/logs/app.log
```

Configure log rotation:

```bash
sudo nano /etc/logrotate.d/piperag
```

Add this content:

```
/home/piperag_admin/piperag/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 piperag_admin piperag_admin
}
```

## Testing the Deployment

Test the API endpoint:

```bash
curl -X GET "https://api.yourdomain.com/health" -H "X-API-Key: your-api-key"
```

You should receive a JSON response with status "healthy".

## Updating Your Flutter Web App

Update your Flutter web app to connect to the API:

1. Build with the production URL and API key:
   ```bash
   flutter build web --release --web-renderer canvaskit --dart-define=API_BASE_URL=https://api.yourdomain.com --dart-define=API_KEY=your-api-key
   ```

2. Deploy to Firebase:
   ```bash
   firebase deploy --only hosting
   ```

## Security Recommendations

1. **Rotate API Keys Regularly**
   - Generate new API keys periodically
   - Update both the server and Flutter app with the new key

2. **Monitor Logs**
   - Regularly check logs for suspicious activities:
   ```bash
   sudo journalctl -u piperag.service
   ```

3. **Automatic Updates**
   - Set up automatic security updates:
   ```bash
   sudo apt install unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades
   ```

4. **Backup**
   - Regularly backup your model files and database

## Troubleshooting

- **Service won't start**
  - Check logs: `sudo journalctl -u piperag.service`
  - Verify environment variables and paths

- **Permission issues**
  - Check ownership: `ls -la /home/piperag_admin/piperag`
  - Fix permissions: `sudo chown -R piperag_admin:piperag_admin /home/piperag_admin/piperag`

- **Model loading problems**
  - Verify model files exist and are correctly referenced in .env
  - Check system has enough RAM for model loading

## Maintenance

- **Updating the API**
  ```bash
  cd /home/piperag_admin/piperag
  git pull
  source venv/bin/activate
  pip install -r requirements.txt
  sudo systemctl restart piperag
  ```

- **Checking service status**
  ```bash
  sudo systemctl status piperag
  ```