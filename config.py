# config.py
import os
import yaml
import logging
from logging.handlers import RotatingFileHandler
import json

logger = logging.getLogger(__name__)

class Configuration:
    """Configuration manager for the anomaly detection system"""
    
    def __init__(self, config_path="./config.yaml"):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create default"""
        # Check if config file exists
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found at {self.config_path}, creating default")
            config = self._create_default_config()
            self._save_config(config)
            return config
        
        try:
            # Load configuration from YAML file
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using default configuration")
            return self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        default_config = {
            'graylog': {
                'api_url': 'https://graylog.example.com/api',
                'api_key': 'your_graylog_api_key',
                'query_interval_minutes': 10
            },
            'database': {
                'path': './embeddings_db',
                'max_age_days': 14
            },
            'model': {
                'embedding_dim': 64,
                'num_clusters': 5,
                'anomaly_threshold': 0.7,
                'path': './model'
            },
            'alerts': {
                'email': {
                    'enabled': True,
                    'recipients': ['admin@datacenter.com'],
                    'smtp_server': 'smtp.example.com',
                    'smtp_port': 587,
                    'smtp_username': '',
                    'smtp_password': '',
                    'sender': 'alerts@datacenter.com',
                    'cooldown_minutes': 30
                },
                'webhook': {
                    'enabled': False,
                    'url': 'https://hooks.slack.com/services/your/webhook/url',
                    'cooldown_minutes': 30
                }
            },
            'logging': {
                'level': 'INFO',
                'file': './logs/anomaly_detector.log',
                'max_size_mb': 10,
                'backup_count': 5
            }
        }
        
        return default_config
    
    def _save_config(self, config):
        """Save configuration to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save configuration to YAML file
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get(self, key, default=None):
        """
        Get a configuration value
        
        Args:
            key: Configuration key (dot-separated for nested values)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        # Traverse nested dictionary
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """
        Set a configuration value
        
        Args:
            key: Configuration key (dot-separated for nested values)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Traverse nested dictionary to second-last key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value at final key
        config[keys[-1]] = value
        
        # Save updated configuration
        self._save_config(self.config)
    
    def update(self, updates):
        """
        Update multiple configuration values
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def setup_logging(self):
        """Set up logging based on configuration"""
        log_level = getattr(logging, self.get('logging.level', 'INFO'))
        log_file = self.get('logging.file', './logs/anomaly_detector.log')
        max_size_mb = self.get('logging.max_size_mb', 10)
        backup_count = self.get('logging.backup_count', 5)
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Configure file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        logger.info("Logging configured")


def load_configuration():
    """Helper function to load configuration"""
    config = Configuration()
    config.setup_logging()
    return config
