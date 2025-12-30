-- Database initialization script for Smart Retail Analytics
-- PostgreSQL with TimescaleDB extension (optional, for time-series optimization)

-- Create database (run as postgres user)
-- CREATE DATABASE retail_analytics;

-- Connect to database
-- \c retail_analytics;

-- Enable TimescaleDB extension (if installed)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Devices table
CREATE TABLE IF NOT EXISTS devices (
    id SERIAL PRIMARY KEY,
    device_key VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    location VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_devices_device_key ON devices(device_key);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    CONSTRAINT fk_device FOREIGN KEY (device_id) REFERENCES devices(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_device_track ON sessions(device_id, track_id);

-- Interactions table (time-series data)
CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    age INTEGER NOT NULL,
    gender VARCHAR(20) NOT NULL,
    emotion VARCHAR(50) NOT NULL,
    ad_id VARCHAR(100) NOT NULL,
    ad_name VARCHAR(200),
    CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_session_timestamp ON interactions(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_ad_id ON interactions(ad_id);

-- Advertisements table
CREATE TABLE IF NOT EXISTS advertisements (
    id SERIAL PRIMARY KEY,
    ad_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    target_age_min INTEGER,
    target_age_max INTEGER,
    target_gender VARCHAR(20),
    target_emotions VARCHAR(200),
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_advertisements_ad_id ON advertisements(ad_id);

-- Optional: Convert interactions to hypertable (TimescaleDB)
-- SELECT create_hypertable('interactions', 'timestamp', if_not_exists => TRUE);

-- Insert sample advertisements
INSERT INTO advertisements (ad_id, name, description, target_age_min, target_age_max, target_gender, target_emotions, priority)
VALUES
    ('coffee_morning', 'Cà phê buổi sáng', 'Quảng cáo cà phê cho buổi sáng', 18, 50, 'all', '["happy", "neutral"]', 8),
    ('lunch_special', 'Ưu đãi trưa', 'Quảng cáo ưu đãi bữa trưa', 20, 60, 'all', '["all"]', 9),
    ('premium_product', 'Sản phẩm cao cấp', 'Quảng cáo sản phẩm cao cấp', 30, 65, 'all', '["happy"]', 7)
ON CONFLICT (ad_id) DO NOTHING;

