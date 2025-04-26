# database.py
import os
import logging
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EmbeddingDatabase:
    """Database for storing event embeddings and metadata"""
    
    def __init__(self, db_path="./embeddings_db", max_age_days=14):
        """
        Initialize the embedding database
        
        Args:
            db_path: Path to the database directory
            max_age_days: Maximum age of events to keep (for pruning)
        """
        self.db_path = db_path
        self.max_age_days = max_age_days
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database and create tables if they don't exist
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database tables"""
        db_file = f"{self.db_path}/embeddings.db"
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Create events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            embedding BLOB NOT NULL,
            event_data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        
        # Create index on timestamp for efficient queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized embedding database at {db_file}")
    
    def _serialize_embedding(self, embedding):
        """Serialize numpy array to bytes for storage"""
        return embedding.tobytes()
    
    def _deserialize_embedding(self, blob, embedding_dim):
        """Deserialize bytes to numpy array"""
        return np.frombuffer(blob, dtype=np.float64).reshape(-1)
    
    def store(self, event_id, embedding, timestamp, event_data):
        """
        Store an event embedding in the database
        
        Args:
            event_id: Unique ID for the event
            embedding: Numpy array of the event embedding
            timestamp: Timestamp of the event
            event_data: Original event data (will be JSON serialized)
        """
        try:
            # Connect to database
            db_file = f"{self.db_path}/embeddings.db"
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Serialize embedding and event data
            embedding_blob = self._serialize_embedding(embedding)
            event_json = json.dumps(event_data)
            
            # Current timestamp
            created_at = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute(
                '''
                INSERT OR REPLACE INTO events
                (id, timestamp, embedding, event_data, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (event_id, timestamp, embedding_blob, event_json, created_at)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing event in database: {e}")
    
    def get_events_in_timeframe(self, start_time, end_time, limit=1000):
        """
        Retrieve events within a specified timeframe
        
        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            limit: Maximum number of events to return
            
        Returns:
            List of (event_id, timestamp, embedding, event_data) tuples
        """
        try:
            # Connect to database
            db_file = f"{self.db_path}/embeddings.db"
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Query events within timeframe
            cursor.execute(
                '''
                SELECT id, timestamp, embedding, event_data
                FROM events
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
                ''',
                (start_time, end_time, limit)
            )
            
            # Process results
            results = []
            for row in cursor.fetchall():
                event_id, timestamp, embedding_blob, event_json = row
                embedding = self._deserialize_embedding(embedding_blob, -1)  # Infer dimension
                event_data = json.loads(event_json)
                results.append((event_id, timestamp, embedding, event_data))
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving events from database: {e}")
            return []
    
    def prune_old_events(self):
        """Remove events older than max_age_days from the database"""
        try:
            # Calculate cutoff timestamp
            cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
            cutoff_timestamp = cutoff_date.isoformat()
            
            # Connect to database
            db_file = f"{self.db_path}/embeddings.db"
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Count events to be pruned
            cursor.execute(
                '''
                SELECT COUNT(*) FROM events
                WHERE timestamp < ?
                ''',
                (cutoff_timestamp,)
            )
            count = cursor.fetchone()[0]
            
            # Delete old events
            if count > 0:
                cursor.execute(
                    '''
                    DELETE FROM events
                    WHERE timestamp < ?
                    ''',
                    (cutoff_timestamp,)
                )
                conn.commit()
                logger.info(f"Pruned {count} events older than {self.max_age_days} days")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error pruning old events: {e}")
    
    def get_all_embeddings(self, limit=10000):
        """
        Retrieve all embeddings from the database
        
        Args:
            limit: Maximum number of embeddings to return
            
        Returns:
            List of embedding vectors
        """
        try:
            # Connect to database
            db_file = f"{self.db_path}/embeddings.db"
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Query embeddings
            cursor.execute(
                '''
                SELECT embedding
                FROM events
                ORDER BY timestamp DESC
                LIMIT ?
                ''',
                (limit,)
            )
            
            # Process results
            embeddings = []
            for (embedding_blob,) in cursor.fetchall():
                embedding = self._deserialize_embedding(embedding_blob, -1)  # Infer dimension
                embeddings.append(embedding)
            
            conn.close()
            return embeddings
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings from database: {e}")
            return []
    
    def get_event_count(self):
        """Get the total number of events in the database"""
        try:
            # Connect to database
            db_file = f"{self.db_path}/embeddings.db"
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Query count
            cursor.execute('SELECT COUNT(*) FROM events')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting event count: {e}")
            return 0
