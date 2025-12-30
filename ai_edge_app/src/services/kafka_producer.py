"""
Kafka Producer Service for Edge-to-Cloud Communication
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """Service for sending data to Kafka topics"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_telemetry: str = "edge-telemetry",
        topic_images: str = "edge-images"
    ):
        """
        Initialize Kafka Producer
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic_telemetry: Topic for telemetry data
            topic_images: Topic for image data
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic_telemetry = topic_telemetry
        self.topic_images = topic_images
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True
            )
            logger.info(f"Kafka Producer initialized: {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            self.producer = None
    
    def send_telemetry(
        self,
        device_id: str,
        age: int,
        gender: str,
        emotion: str,
        confidence: float,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Send telemetry data to Kafka
        
        Args:
            device_id: Edge device identifier
            age: Detected age
            gender: Detected gender
            emotion: Detected emotion
            confidence: Detection confidence
            timestamp: Event timestamp
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.producer:
            logger.warning("Kafka Producer not initialized")
            return False
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        data = {
            "device_id": device_id,
            "timestamp": timestamp.isoformat(),
            "demographics": {
                "age": age,
                "gender": gender,
                "emotion": emotion
            },
            "confidence": confidence,
            "event_type": "customer_detection"
        }
        
        try:
            future = self.producer.send(
                self.topic_telemetry,
                key=device_id,
                value=data
            )
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            logger.debug(
                f"Telemetry sent: topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}"
            )
            return True
        except KafkaError as e:
            logger.error(f"Failed to send telemetry: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending telemetry: {e}")
            return False
    
    def send_image_metadata(
        self,
        device_id: str,
        image_path: str,
        minio_bucket: str = "retail-data",
        minio_path: str = None,
        reason: str = "low_confidence"
    ) -> bool:
        """
        Send image metadata to Kafka (image uploaded to MinIO)
        
        Args:
            device_id: Edge device identifier
            image_path: Local path to image
            minio_bucket: MinIO bucket name
            minio_path: Path in MinIO
            reason: Reason for uploading (low_confidence, error, etc.)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.producer:
            logger.warning("Kafka Producer not initialized")
            return False
        
        data = {
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat(),
            "image": {
                "bucket": minio_bucket,
                "path": minio_path or f"{device_id}/{datetime.utcnow().strftime('%Y%m%d')}/{image_path}",
                "local_path": image_path
            },
            "reason": reason,
            "event_type": "image_upload"
        }
        
        try:
            future = self.producer.send(
                self.topic_images,
                key=device_id,
                value=data
            )
            record_metadata = future.get(timeout=10)
            logger.info(f"Image metadata sent: {minio_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to send image metadata: {e}")
            return False
    
    def flush(self):
        """Flush all pending messages"""
        if self.producer:
            self.producer.flush()
    
    def close(self):
        """Close the producer"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka Producer closed")

