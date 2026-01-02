"""
Test admin login
"""
from app.core.database import SessionLocal
from app.models.user import User
from app.core.security import verify_password

db = SessionLocal()
try:
    admin = db.query(User).filter(User.username == 'admin').first()
    if admin:
        print(f'[OK] Admin found: {admin.username}')
        result = verify_password('admin123', admin.hashed_password)
        print(f'[OK] Password verify: {result}')
    else:
        print('[ERROR] Admin NOT FOUND')
finally:
    db.close()
