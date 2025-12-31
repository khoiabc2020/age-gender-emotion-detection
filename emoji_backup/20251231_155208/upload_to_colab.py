"""
Script tự động upload code lên Google Drive để sử dụng trên Colab
"""

import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    import pickle
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print("⚠️  Cần cài đặt: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")


def create_zip_file(source_dir, output_zip):
    """Tạo file zip từ thư mục"""
    print(f"📦 Đang tạo file zip từ {source_dir}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Bỏ qua các thư mục không cần thiết
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.git', 'venv', 'venv_gpu', 
                'node_modules', '.pytest_cache', '.ipynb_checkpoints',
                'checkpoints', 'results', 'logs', 'data'
            ]]
            
            for file in files:
                # Bỏ qua các file không cần thiết
                if file.endswith(('.pyc', '.pyo', '.pyd', '.log', '.tmp')):
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"✅ Đã tạo file zip: {output_zip} ({size_mb:.2f} MB)")
    return output_zip


def authenticate_google_drive():
    """Xác thực với Google Drive API"""
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None
    
    # File token lưu credentials
    token_file = 'token.pickle'
    creds_file = 'credentials.json'
    
    # Load credentials đã lưu
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # Nếu không có credentials hợp lệ, yêu cầu đăng nhập
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_file):
                print("❌ Không tìm thấy credentials.json")
                print("\n📝 Cách lấy credentials.json:")
                print("1. Truy cập: https://console.cloud.google.com/")
                print("2. Tạo project mới (hoặc chọn project có sẵn)")
                print("3. Enable Google Drive API:")
                print("   - APIs & Services → Enable APIs → Google Drive API")
                print("4. Tạo OAuth 2.0 credentials:")
                print("   - APIs & Services → Credentials → Create Credentials → OAuth client ID")
                print("   - Application type: Desktop app")
                print("   - Download và lưu thành credentials.json")
                print("5. Đặt credentials.json vào thư mục này")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Lưu credentials cho lần sau
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds


def upload_to_drive(file_path, folder_name='Colab_Training', service=None):
    """Upload file lên Google Drive"""
    if not GOOGLE_DRIVE_AVAILABLE:
        print("❌ Google Drive API chưa được cài đặt")
        return None
    
    if service is None:
        creds = authenticate_google_drive()
        if not creds:
            return None
        service = build('drive', 'v3', credentials=creds)
    
    # Tìm hoặc tạo thư mục
    folder_id = None
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if items:
        folder_id = items[0]['id']
        print(f"✅ Tìm thấy thư mục: {folder_name}")
    else:
        # Tạo thư mục mới
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        folder_id = folder.get('id')
        print(f"✅ Đã tạo thư mục mới: {folder_name}")
    
    # Upload file
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(file_path, resumable=True)
    print(f"📤 Đang upload {file_name}...")
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    file_id = file.get('id')
    file_link = file.get('webViewLink')
    
    print(f"✅ Upload thành công!")
    print(f"   File ID: {file_id}")
    print(f"   Link: {file_link}")
    
    return file_id, file_link


def main():
    """Hàm chính"""
    print("=" * 60)
    print("🚀 Tự động upload code lên Google Drive cho Colab")
    print("=" * 60)
    
    # Đường dẫn thư mục training_experiments
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    training_dir = project_root / 'training_experiments'
    
    if not training_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {training_dir}")
        return
    
    # Tạo file zip
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_file = project_root / f'training_experiments_{timestamp}.zip'
    
    try:
        create_zip_file(str(training_dir), str(zip_file))
        
        # Upload lên Google Drive
        if GOOGLE_DRIVE_AVAILABLE:
            print("\n" + "=" * 60)
            print("📤 Upload lên Google Drive...")
            print("=" * 60)
            
            result = upload_to_drive(str(zip_file))
            
            if result:
                file_id, file_link = result
                print("\n" + "=" * 60)
                print("✅ HOÀN TẤT!")
                print("=" * 60)
                print(f"\n📁 File đã được upload lên Google Drive")
                print(f"🔗 Link: {file_link}")
                print(f"\n📝 Các bước tiếp theo:")
                print(f"1. Mở Google Colab: https://colab.research.google.com/")
                print(f"2. Mount Google Drive trong Colab")
                print(f"3. Giải nén file zip từ Drive")
                print(f"4. Chạy notebook train_on_colab.ipynb")
        else:
            print("\n" + "=" * 60)
            print("📦 File zip đã được tạo")
            print("=" * 60)
            print(f"📁 Vị trí: {zip_file}")
            print(f"\n📝 Các bước tiếp theo:")
            print(f"1. Upload file zip lên Google Drive thủ công")
            print(f"2. Mở Google Colab và mount Drive")
            print(f"3. Giải nén file zip")
            print(f"4. Chạy notebook train_on_colab.ipynb")
    
    finally:
        # Xóa file zip tạm (tùy chọn)
        # os.remove(zip_file)
        pass


if __name__ == "__main__":
    main()


