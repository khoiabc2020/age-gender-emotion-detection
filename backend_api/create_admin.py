"""
Create default admin user
"""
from app.core.database import SessionLocal, engine, Base
from app.models.user import User
from app.core.security import get_password_hash

# Create tables
Base.metadata.create_all(bind=engine)

# Create admin user
db = SessionLocal()
try:
    admin = db.query(User).filter(User.username == 'admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@retail.com',
            hashed_password=get_password_hash('admin123'),
            full_name='Administrator',
            is_active=True,
            is_superuser=True
        )
        db.add(admin)
        db.commit()
        print('✅ Admin user created successfully!')
        print('   Username: admin')
        print('   Password: admin123')
    else:
        print('✅ Admin user already exists')
finally:
    db.close()
