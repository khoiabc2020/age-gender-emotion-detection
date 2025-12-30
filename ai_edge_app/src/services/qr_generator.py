import qrcode
import cv2
import numpy as np
import time
from typing import Optional, Tuple

class QRCodeService:
    """
    Service to generate dynamic QR codes for interactive ads.
    """
    
    def __init__(self):
        self.cache = {}
        
    def generate_qr(self, data: str, size: int = 150) -> np.ndarray:
        """
        Generate a QR code image
        
        Args:
            data: Content of the QR code (URL or Voucher Code)
            size: Target size in pixels (square)
            
        Returns:
            OpenCV image (BGR format)
        """
        if data in self.cache:
            return self.cache[data]
            
        # Generate QR
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Convert to image
        img_pil = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL to Numpy
        img_np = np.array(img_pil.convert('RGB'))
        
        # Convert RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Resize to target size
        img_resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        
        # Cache current QR (simple cache)
        if len(self.cache) > 20:
            self.cache.clear() # Clear if too big
        self.cache[data] = img_resized
        
        return img_resized
        
    def get_voucher_qr(self, ad_id: str) -> Tuple[np.ndarray, str]:
        """
        Generate a unique voucher QR for an ad
        """
        timestamp = int(time.time())
        voucher_code = f"DISCOUNT_{ad_id.upper()}_{timestamp}" # Simple unique code
        # In a real app, this URL would point to a redemption page
        url = f"https://mystore.com/redeem?code={voucher_code}"
        
        qr_img = self.generate_qr(url)
        return qr_img, "20% OFF"
