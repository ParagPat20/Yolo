#!/bin/bash

# Simple FTP Server Setup - No Security, Just Works!
# For local projects only - NOT for production use
# Usage: sudo ./simple_ftp_server.sh [recordings_directory]

RECORDINGS_DIR=${1:-"recordings"}

echo "========================================="
echo "Simple FTP Server Setup (No Security)"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Install vsftpd if not already installed
echo "Installing FTP server..."
apt update
apt install -y vsftpd

# Stop the service first
systemctl stop vsftpd

# Create recordings directory with open permissions
mkdir -p "/home/jecon/$RECORDINGS_DIR"
chmod 777 "/home/jecon/$RECORDINGS_DIR"

# Create a very simple vsftpd config - NO SECURITY!
cat > /etc/vsftpd.conf << 'EOF'
# Simple FTP server - NO SECURITY!
listen=YES
listen_ipv6=NO

# Allow anonymous access - NO PASSWORD NEEDED!
anonymous_enable=YES
anon_upload_enable=YES
anon_mkdir_write_enable=YES
anon_other_write_enable=YES
anon_world_readable_only=NO
anon_root=/home/jecon

# Also allow local users
local_enable=YES
write_enable=YES
local_umask=000

# No chroot - keep it simple
chroot_local_user=NO

# Passive mode
pasv_enable=YES
pasv_min_port=10000
pasv_max_port=10100

# Logging
xferlog_enable=YES
xferlog_file=/var/log/vsftpd.log

# No timeouts
idle_session_timeout=0
data_connection_timeout=0

# Allow everything
connect_from_port_20=YES
use_localtime=YES

# Simple banner
ftpd_banner=Simple FTP Server - No Password Needed!

# Disable security features for simplicity
seccomp_sandbox=NO
allow_anon_ssl=NO
force_local_data_ssl=NO
force_local_logins_ssl=NO

# File permissions
file_open_mode=0666
local_umask=000
anon_umask=000
EOF

# Get Pi IP
PI_IP=$(hostname -I | awk '{print $1}')

# Disable firewall for FTP (if ufw is active)
if command -v ufw &> /dev/null; then
    echo "Disabling firewall for FTP..."
    ufw allow 21/tcp
    ufw allow 10000:10100/tcp
fi

# Start and enable the service
systemctl enable vsftpd
systemctl start vsftpd

# Wait a moment for service to start
sleep 2

# Check if service is running
if systemctl is-active --quiet vsftpd; then
    echo ""
    echo "========================================="
    echo "✅ Simple FTP Server is Running!"
    echo "========================================="
    echo "Server IP: $PI_IP"
    echo "Port: 21"
    echo "Username: anonymous (or just press Enter)"
    echo "Password: (none - just press Enter)"
    echo ""
    echo "📁 Files location: /home/jecon/$RECORDINGS_DIR"
    echo ""
    echo "🔗 Connection Examples:"
    echo "----------------------------------------"
    echo "Windows Command Line:"
    echo "ftp $PI_IP"
    echo "Username: anonymous"
    echo "Password: (just press Enter)"
    echo ""
    echo "Python:"
    echo "from ftplib import FTP"
    echo "ftp = FTP('$PI_IP')"
    echo "ftp.login()  # No username/password needed"
    echo "ftp.retrlines('LIST')"
    echo ""
    echo "Your download command:"
    echo "python downndet.py --host 192.168.29.235 --detection-only downloaded_videos --model best.pt --classes "17,19,27,42,51,55,80""
    echo "========================================="
else
    echo "❌ Failed to start FTP server"
    echo "Check logs: sudo journalctl -u vsftpd -f"
    exit 1
fi

# Create a simple test file
echo "FTP Server Test File - $(date)" > "/home/jecon/$RECORDINGS_DIR/test.txt"
chmod 666 "/home/jecon/$RECORDINGS_DIR/test.txt"

echo ""
echo "✅ Setup complete! Test file created at /home/jecon/$RECORDINGS_DIR/test.txt"
