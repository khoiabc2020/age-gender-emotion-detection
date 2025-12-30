"""
MQTT Worker for processing analytics data from edge devices
"""

import json
import logging
from typing import Dict
import paho.mqtt.client as mqtt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.db import crud
from app.schemas.analytics import InteractionCreate

logger = logging.getLogger(__name__)


class MQTTWorker:
    """MQTT worker to process messages from edge devices"""
    
    def __init__(self):
        self.client = None
        self.db: Session = SessionLocal()
    
    def connect(self):
        """Connect to MQTT broker"""
        self.client = mqtt.Client(client_id="backend_worker")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(settings.MQTT_BROKER, settings.MQTT_PORT, keepalive=60)
        self.client.loop_start()
        logger.info(f"MQTT Worker connected to {settings.MQTT_BROKER}:{settings.MQTT_PORT}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for connection"""
        if rc == 0:
            client.subscribe(settings.MQTT_TOPIC)
            logger.info(f"Subscribed to topic: {settings.MQTT_TOPIC}")
        else:
            logger.error(f"Failed to connect, return code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT message"""
        try:
            payload = json.loads(msg.payload.decode())
            logger.debug(f"Received message: {payload}")
            
            # Create interaction record
            interaction = InteractionCreate(
                device_key=payload.get('device_key'),
                track_id=payload.get('track_id'),
                age=payload.get('age'),
                gender=payload.get('gender'),
                emotion=payload.get('emotion'),
                ad_id=payload.get('ad_id')
            )
            
            # Save to database
            device = crud.get_device_by_key(self.db, interaction.device_key)
            if not device:
                device = crud.create_device(self.db, interaction.device_key)
            
            session = crud.get_active_session(self.db, device.id, interaction.track_id)
            if not session:
                session = crud.create_session(self.db, device.id, interaction.track_id)
            
            crud.create_interaction(
                db=self.db,
                session_id=session.id,
                age=interaction.age,
                gender=interaction.gender,
                emotion=interaction.emotion,
                ad_id=interaction.ad_id
            )
            
            logger.info(f"Processed interaction for track_id: {interaction.track_id}")
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        if self.db:
            self.db.close()


def run_worker():
    """Run MQTT worker"""
    worker = MQTTWorker()
    try:
        worker.connect()
        # Keep running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping MQTT worker...")
        worker.disconnect()


if __name__ == "__main__":
    run_worker()

