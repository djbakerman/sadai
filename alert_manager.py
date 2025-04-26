# alert_manager.py
# made with love

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailAlertManager:
    """Manager for sending email alerts when anomalies are detected"""
    
    def __init__(self, recipients, 
                 smtp_server=None, 
                 smtp_port=None,
                 smtp_username=None,
                 smtp_password=None,
                 sender=None,
                 cooldown_minutes=30):
        """
        Initialize the email alert manager
        
        Args:
            recipients: List of email addresses to receive alerts
            smtp_server: SMTP server address (default: from env var SMTP_SERVER)
            smtp_port: SMTP server port (default: from env var SMTP_PORT)
            smtp_username: SMTP username (default: from env var SMTP_USERNAME)
            smtp_password: SMTP password (default: from env var SMTP_PASSWORD)
            sender: Sender email address (default: from env var ALERT_SENDER)
            cooldown_minutes: Minimum minutes between alerts to prevent spamming
        """
        # Email recipients
        self.recipients = recipients if isinstance(recipients, list) else [recipients]
        
        # SMTP configuration (from environment variables or parameters)
        self.smtp_server = smtp_server or os.environ.get('SMTP_SERVER', 'smtp.example.com')
        self.smtp_port = int(smtp_port or os.environ.get('SMTP_PORT', '587'))
        self.smtp_username = smtp_username or os.environ.get('SMTP_USERNAME', '')
        self.smtp_password = smtp_password or os.environ.get('SMTP_PASSWORD', '')
        self.sender = sender or os.environ.get('ALERT_SENDER', 'alerts@datacenter.com')
        
        # Cooldown to prevent alert fatigue
        self.cooldown_minutes = cooldown_minutes
        self.last_alert_time = None
    
    def _is_on_cooldown(self):
        """Check if alerts are currently on cooldown"""
        if self.last_alert_time is None:
            return False
            
        now = datetime.now()
        elapsed_minutes = (now - self.last_alert_time).total_seconds() / 60
        
        return elapsed_minutes < self.cooldown_minutes
    
    def send_alert(self, subject, body):
        """
        Send an email alert
        
        Args:
            subject: Email subject
            body: Email body
            
        Returns:
            Boolean indicating success
        """
        # Check cooldown
        if self._is_on_cooldown():
            logger.info(f"Alert on cooldown, skipping (last alert sent {self.last_alert_time})")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            
            # Attach body
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                # Use TLS if required
                if self.smtp_port == 587:
                    server.starttls()
                
                # Login if credentials provided
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                # Send email
                server.send_message(msg)
            
            # Update last alert time
            self.last_alert_time = datetime.now()
            
            logger.info(f"Alert email sent to {', '.join(self.recipients)}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert email: {e}")
            
            # Try with console fallback
            self._console_fallback(subject, body)
            return False
    
    def _console_fallback(self, subject, body):
        """Print alert to console as fallback if email fails"""
        logger.warning("==== ALERT FALLBACK (EMAIL FAILED) ====")
        logger.warning(f"Subject: {subject}")
        logger.warning(f"To: {', '.join(self.recipients)}")
        logger.warning("Message:")
        for line in body.split('\n'):
            logger.warning(line)
        logger.warning("=======================================")


class WebhookAlertManager:
    """Manager for sending alerts via webhooks (e.g., Slack, Teams, etc.)"""
    
    def __init__(self, webhook_url, cooldown_minutes=30):
        """
        Initialize the webhook alert manager
        
        Args:
            webhook_url: URL for the webhook
            cooldown_minutes: Minimum minutes between alerts to prevent spamming
        """
        self.webhook_url = webhook_url
        self.cooldown_minutes = cooldown_minutes
        self.last_alert_time = None
    
    def _is_on_cooldown(self):
        """Check if alerts are currently on cooldown"""
        if self.last_alert_time is None:
            return False
            
        now = datetime.now()
        elapsed_minutes = (now - self.last_alert_time).total_seconds() / 60
        
        return elapsed_minutes < self.cooldown_minutes
    
    def send_alert(self, subject, body):
        """
        Send a webhook alert
        
        Args:
            subject: Alert subject
            body: Alert body
            
        Returns:
            Boolean indicating success
        """
        # Check cooldown
        if self._is_on_cooldown():
            logger.info(f"Alert on cooldown, skipping (last alert sent {self.last_alert_time})")
            return False
        
        try:
            import requests
            
            # Format payload for webhook
            # This is a generic format - adjust based on your webhook provider
            payload = {
                "title": subject,
                "text": body,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Check response
            response.raise_for_status()
            
            # Update last alert time
            self.last_alert_time = datetime.now()
            
            logger.info(f"Webhook alert sent, status code: {response.status_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            
            # Try with console fallback
            self._console_fallback(subject, body)
            return False
    
    def _console_fallback(self, subject, body):
        """Print alert to console as fallback if webhook fails"""
        logger.warning("==== ALERT FALLBACK (WEBHOOK FAILED) ====")
        logger.warning(f"Subject: {subject}")
        logger.warning("Message:")
        for line in body.split('\n'):
            logger.warning(line)
        logger.warning("==========================================")
