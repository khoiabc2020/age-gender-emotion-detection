"""
MQTT Client for async data transmission to cloud
"""

import json
import paho.mqtt.client as mqtt
from typing import Dict, Optional
import logging

class MQTTClient:
    """
    MQTT client for sending analytics data to cloud server
    """
    
    def __init__(
        self, 
        broker: str, 
        port: int, 
        topic: str, 
        device_key: str
    ):
        """
        Initialize MQTT client
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            topic: Topic to publish to
            device_key: Device authentication key
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.device_key = device_key
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client = mqtt.Client(client_id=f"edge_device_{self.device_key}")
            self.client.on_connect = self._on_connect
            self.client.on_publish = self._on_publish
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def publish_analytics(
        self, 
        track_id: int, 
        age: int, 
        gender: str, 
        emotion: str,
        ad_id: str
    ):
        """
        Publish analytics data to MQTT broker
        
        Args:
            track_id: Tracking ID
            age: Customer age
            gender: Customer gender
            emotion: Customer emotion
            ad_id: Displayed advertisement ID
        """
        if not self.client or not self.client.is_connected():
            self.logger.warning("MQTT client not connected, skipping publish")
            return
        
        payload = {
            'device_key': self.device_key,
            'track_id': track_id,
            'timestamp': None,  # Will be set by backend
            'age': age,
            'gender': gender,
            'emotion': emotion,
            'ad_id': ad_id
        }
        
        try:
            message = json.dumps(payload)
            result = self.client.publish(self.topic, message, qos=1)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                self.logger.error(f"Failed to publish message: {result.rc}")
        except Exception as e:
            self.logger.error(f"Error publishing analytics: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for connection"""
        if rc == 0:
            self.logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
        else:
            self.logger.error(f"Failed to connect, return code {rc}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for publish"""
        self.logger.debug(f"Message published with mid: {mid}")

