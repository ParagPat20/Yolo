#!/bin/bash

# Raspberry Pi FTP Server Setup Script
# Sets up vsftpd FTP server to share video recordings
# Usage: ./setup_ftp_server.sh [recordings_directory] [ftp_username]

# Configuration
RECORDINGS_DIR=${1:-"recordings"}
FTP_USERNAME=${2:-"videouser"}
FTP_PASSWORD=""
FTP_PORT=21
PASSIVE_MIN_PORT=10000
PASSIVE_MAX_PORT=10100

echo "========================================="
echo "Raspberry Pi FTP Server Setup"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Install vsftpd if not already installed
echo "Installing FTP server (vsftpd)..."
apt update
apt install -y vsftpd

# Create FTP user if it doesn't exist
if ! id "$FTP_USERNAME" &>/dev/null; then
    echo "Creating FTP user: $FTP_USERNAME"
    
    # Generate random password
    FTP_PASSWORD=$(openssl rand -base64 12)
    
    # Create user with home directory
    useradd -m -d /home/$FTP_USERNAME -s /bin/bash $FTP_USERNAME
    echo "$FTP_USERNAME:$FTP_PASSWORD" | chpasswd
    
    echo "User created with password: $FTP_PASSWORD"
    echo "IMPORTANT: Save this password!"
else
    echo "User $FTP_USERNAME already exists"
    echo "To reset password, run: sudo passwd $FTP_USERNAME"
fi

# Create recordings directory if it doesn't exist
FULL_RECORDINGS_PATH="/home/$FTP_USERNAME/$RECORDINGS_DIR"
mkdir -p "$FULL_RECORDINGS_PATH"
chown $FTP_USERNAME:$FTP_USERNAME "$FULL_RECORDINGS_PATH"

# Create a symlink to the actual recordings directory if it's elsewhere
CURRENT_RECORDINGS_PATH=$(pwd)/$RECORDINGS_DIR
if [ -d "$CURRENT_RECORDINGS_PATH" ] && [ "$CURRENT_RECORDINGS_PATH" != "$FULL_RECORDINGS_PATH" ]; then
    echo "Creating symlink to existing recordings..."
    ln -sf "$CURRENT_RECORDINGS_PATH" "$FULL_RECORDINGS_PATH"
fi

# Backup original vsftpd config
cp /etc/vsftpd.conf /etc/vsftpd.conf.backup

# Create new vsftpd configuration
cat > /etc/vsftpd.conf << EOF
# Basic FTP server configuration for video sharing
listen=YES
listen_ipv6=NO

# Anonymous access
anonymous_enable=NO

# Local user access
local_enable=YES
write_enable=YES
local_umask=022

# Security
chroot_local_user=YES
allow_writeable_chroot=YES
secure_chroot_dir=/var/run/vsftpd/empty

# Passive mode configuration
pasv_enable=YES
pasv_min_port=$PASSIVE_MIN_PORT
pasv_max_port=$PASSIVE_MAX_PORT
pasv_address=

# Logging
xferlog_enable=YES
xferlog_file=/var/log/vsftpd.log

# Performance
use_localtime=YES
connect_from_port_20=YES

# User restrictions
userlist_enable=YES
userlist_file=/etc/vsftpd.userlist
userlist_deny=NO

# File permissions
file_open_mode=0666
local_root=/home/$FTP_USERNAME

# Connection limits
max_clients=10
max_per_ip=3

# Timeout settings
idle_session_timeout=300
data_connection_timeout=120

# Banner
ftpd_banner=Video Recording FTP Server
EOF

# Add FTP user to allowed users list
echo "$FTP_USERNAME" > /etc/vsftpd.userlist

# Get Raspberry Pi IP address
PI_IP=$(hostname -I | awk '{print $1}')

# Enable and start vsftpd service
systemctl enable vsftpd
systemctl restart vsftpd

# Configure firewall (if ufw is installed)
if command -v ufw &> /dev/null; then
    echo "Configuring firewall..."
    ufw allow $FTP_PORT/tcp
    ufw allow $PASSIVE_MIN_PORT:$PASSIVE_MAX_PORT/tcp
fi

# Create a simple client connection script
cat > /home/$FTP_USERNAME/ftp_connect_info.txt << EOF
========================================
FTP Server Connection Information
========================================

Server IP: $PI_IP
Port: $FTP_PORT
Username: $FTP_USERNAME
Password: $FTP_PASSWORD

Recordings Directory: $RECORDINGS_DIR

========================================
Connection Examples:
========================================

Windows Command Line:
ftp $PI_IP

FileZilla:
Host: ftp://$PI_IP
Username: $FTP_USERNAME
Password: $FTP_PASSWORD
Port: $FTP_PORT

Linux/Mac Terminal:
ftp $PI_IP
# or
lftp ftp://$FTP_USERNAME:$FTP_PASSWORD@$PI_IP

Python Script:
from ftplib import FTP
ftp = FTP('$PI_IP')
ftp.login('$FTP_USERNAME', '$FTP_PASSWORD')
ftp.retrlines('LIST')

========================================
EOF

chown $FTP_USERNAME:$FTP_USERNAME /home/$FTP_USERNAME/ftp_connect_info.txt

echo ""
echo "========================================="
echo "FTP Server Setup Complete!"
echo "========================================="
echo "Server IP: $PI_IP"
echo "FTP Port: $FTP_PORT"
echo "Username: $FTP_USERNAME"
if [ ! -z "$FTP_PASSWORD" ]; then
    echo "Password: $FTP_PASSWORD"
fi
echo "Recordings Directory: $RECORDINGS_DIR"
echo ""
echo "Connection info saved to: /home/$FTP_USERNAME/ftp_connect_info.txt"
echo ""
echo "Test connection:"
echo "ftp $PI_IP"
echo ""
echo "To check server status: systemctl status vsftpd"
echo "To view logs: tail -f /var/log/vsftpd.log"
echo "========================================="
