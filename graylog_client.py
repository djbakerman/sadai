# graylog_client.py
import logging
import requests
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class GraylogClient:
    """Client for interacting with the Graylog API to fetch syslog events"""
    
    def __init__(self, api_url, api_key):
        """
        Initialize the Graylog API client
        
        Args:
            api_url: Base URL for the Graylog API
            api_key: API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def query_syslog(self, from_time, to_time, limit=1000):
        """
        Query the Graylog API for syslog events within a time range
        
        Args:
            from_time: Start time (datetime object)
            to_time: End time (datetime object)
            limit: Maximum number of events to return
            
        Returns:
            List of syslog events
        """
        # Format timestamps for Graylog API
        from_timestamp = self._format_timestamp(from_time)
        to_timestamp = self._format_timestamp(to_time)
        
        # Build the search query
        query_params = {
            'query': '*',  # Fetch all events
            'from': from_timestamp,
            'to': to_timestamp,
            'limit': limit,
            'fields': 'timestamp,message,source,level'  # Specify fields to return
        }
        
        try:
            # Make the request to the search API
            search_url = f"{self.api_url}/search/universal/absolute"
            response = requests.get(
                search_url, 
                headers=self.headers, 
                params=query_params,
                timeout=30
            )
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract messages from the response
            if 'messages' in result:
                # Transform the message format to a simpler structure
                events = [self._transform_event(msg) for msg in result['messages']]
                logger.info(f"Retrieved {len(events)} events from Graylog")
                return events
            else:
                logger.warning("No messages found in Graylog response")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Graylog API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Graylog API response: {e}")
            return []
    
    def _format_timestamp(self, dt):
        """Format datetime object for Graylog API (ISO8601)"""
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
    def _transform_event(self, graylog_message):
        """
        Transform Graylog message format to a simpler structure
        
        Args:
            graylog_message: Raw message from Graylog API
            
        Returns:
            Simplified event dictionary
        """
        # Extract message fields
        message = graylog_message.get('message', {})
        
        # Create a simplified event structure
        event = {
            'id': message.get('_id', ''),
            'timestamp': message.get('timestamp', ''),
            'message': message.get('message', ''),
            'source': message.get('source', ''),
            'level': message.get('level', 'INFO'),
            'raw': message  # Keep the raw message for reference
        }
        
        return event
