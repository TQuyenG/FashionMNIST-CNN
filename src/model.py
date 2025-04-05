"""
File: model.py
Mục đích: Dùng để định nghĩa kiến trúc mô hình CNN cho Fashion-MNIST.
- Cho file nào/cái gì? Cung cấp mô hình cho `train.py` để huấn luyện và `evaluate.py` để dự đoán.

Quy trình:
1. Khởi tạo mô hình Sequential (xếp tầng tuần tự).
2. Thêm các tầng:
   - Tầng Convolution: Trích xuất đặc trưng từ ảnh.
   - Tầng MaxPooling: Giảm kích thước không gian.
   - Tầng Flatten: Chuyển ma trận thành vector.
   - Tầng Dense: Học đặc trưng cấp cao và phân loại.
   - Tầng Dropout: Ngăn overfitting.
3. Trả về mô hình đã định nghĩa (chưa huấn luyện).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    # Bước 1: Khởi tạo mô hình Sequential
    model = Sequential()
    
    # Bước 2: Thêm các tầng
    # Tầng Convolution 1: Trích xuất đặc trưng cơ bản (đường viền, họa tiết)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # - 32 filter: Tạo 32 bản đồ đặc trưng
    # - (3, 3): Kernel 3x3 quét qua ảnh
    # - relu: Hàm kích hoạt phi tuyến
    # - input_shape: Kích thước ảnh đầu vào
    
    # Tầng MaxPooling 1: Giảm kích thước từ 26x26 xuống 13x13
    model.add(MaxPooling2D((2, 2)))
    
    # Tầng Convolution 2: Trích xuất đặc trưng phức tạp hơn
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # Tầng MaxPooling 2: Giảm kích thước từ 11x11 xuống 5x5
    model.add(MaxPooling2D((2, 2)))
    
    # Tầng Flatten: Chuyển từ (5, 5, 64) thành vector 1600 phần tử
    model.add(Flatten())
    
    # Tầng Dense 1: Học mối quan hệ giữa các đặc trưng
    model.add(Dense(128, activation='relu'))
    
    # Tầng Dropout: Loại bỏ 50% nơ-ron ngẫu nhiên để tránh overfitting
    model.add(Dropout(0.5))
    
    # Tầng Dense 2: Đầu ra 10 lớp, dùng softmax để tính xác suất
    model.add(Dense(10, activation='softmax'))
    
    # Bước 3: Trả về mô hình
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()  # In cấu trúc mô hình để kiểm tra