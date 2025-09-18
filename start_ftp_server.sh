#!/bin/bash

# Simple FTP Server Start/Stop/Status Script
# Usage: ./start_ftp_server.sh [start|stop|restart|status]

ACTION=${1:-"status"}

case $ACTION in
    "start")
        echo "Starting FTP server..."
        sudo systemctl start vsftpd
        if [ $? -eq 0 ]; then
            echo "✓ FTP server started successfully"
            PI_IP=$(hostname -I | awk '{print $1}')
            echo "Server available at: ftp://$PI_IP"
        else
            echo "✗ Failed to start FTP server"
            exit 1
        fi
        ;;
        
    "stop")
        echo "Stopping FTP server..."
        sudo systemctl stop vsftpd
        if [ $? -eq 0 ]; then
            echo "✓ FTP server stopped"
        else
            echo "✗ Failed to stop FTP server"
            exit 1
        fi
        ;;
        
    "restart")
        echo "Restarting FTP server..."
        sudo systemctl restart vsftpd
        if [ $? -eq 0 ]; then
            echo "✓ FTP server restarted successfully"
            PI_IP=$(hostname -I | awk '{print $1}')
            echo "Server available at: ftp://$PI_IP"
        else
            echo "✗ Failed to restart FTP server"
            exit 1
        fi
        ;;
        
    "status")
        echo "FTP Server Status:"
        echo "=================="
        
        # Check if vsftpd is installed
        if ! command -v vsftpd &> /dev/null; then
            echo "✗ vsftpd is not installed"
            echo "Run: sudo ./setup_ftp_server.sh"
            exit 1
        fi
        
        # Check service status
        if systemctl is-active --quiet vsftpd; then
            echo "✓ FTP server is running"
            
            # Show connection info
            PI_IP=$(hostname -I | awk '{print $1}')
            echo "Server IP: $PI_IP"
            echo "Port: 21"
            
            # Show active connections
            CONNECTIONS=$(netstat -an | grep :21 | grep ESTABLISHED | wc -l)
            echo "Active connections: $CONNECTIONS"
            
            # Show last few log entries
            echo ""
            echo "Recent activity:"
            tail -5 /var/log/vsftpd.log 2>/dev/null || echo "No recent activity"
            
        else
            echo "✗ FTP server is not running"
            echo "Start with: ./start_ftp_server.sh start"
        fi
        
        # Check if enabled for auto-start
        if systemctl is-enabled --quiet vsftpd; then
            echo "✓ Auto-start enabled"
        else
            echo "⚠ Auto-start disabled"
        fi
        ;;
        
    "logs")
        echo "FTP Server Logs:"
        echo "================"
        tail -20 /var/log/vsftpd.log 2>/dev/null || echo "No logs found"
        ;;
        
    "info")
        echo "FTP Connection Information:"
        echo "=========================="
        PI_IP=$(hostname -I | awk '{print $1}')
        echo "Server IP: $PI_IP"
        echo "Port: 21"
        echo ""
        echo "Connection examples:"
        echo "ftp $PI_IP"
        echo "lftp ftp://videouser@$PI_IP"
        echo ""
        if [ -f "/home/videouser/ftp_connect_info.txt" ]; then
            echo "Full connection info:"
            cat /home/videouser/ftp_connect_info.txt
        fi
        ;;
        
    *)
        echo "Usage: $0 [start|stop|restart|status|logs|info]"
        echo ""
        echo "Commands:"
        echo "  start   - Start the FTP server"
        echo "  stop    - Stop the FTP server"
        echo "  restart - Restart the FTP server"
        echo "  status  - Show server status (default)"
        echo "  logs    - Show recent log entries"
        echo "  info    - Show connection information"
        exit 1
        ;;
esac
