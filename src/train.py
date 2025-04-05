"""
File: train.py
Mục đích: Dùng để huấn luyện mô hình CNN trên dữ liệu Fashion-MNIST.
- Cho file nào/cái gì? Tạo file mô hình .h5 cho `evaluate.py` sử dụng.

Quy trình:
1. Tải dữ liệu đã xử lý từ data.py (hoặc từ file .npy nếu đã lưu).
2. Khởi tạo mô hình CNN từ model.py.
3. Biên dịch mô hình:
   - Chọn optimizer (Adam), hàm mất mát (categorical_crossentropy), và metric (accuracy).
4. Huấn luyện mô hình:
   - Duyệt qua dữ liệu huấn luyện theo epoch và batch.
   - Cập nhật trọng số dựa trên gradient descent.
   - Đánh giá trên tập kiểm tra sau mỗi epoch.
5. Lưu mô hình đã huấn luyện thành file .h5.
"""

from src.data import load_and_preprocess_data, load_from_files
from src.model import build_model
import tensorflow as tf
import os
import pickle

def train(use_saved_data=False):
    # Bước 1: Tải dữ liệu
    if use_saved_data:
        (x_train, y_train), (x_test, y_test) = load_from_files()
    else:
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(save_files=True)
    
    # Bước 2: Khởi tạo mô hình
    model = build_model()
    
    # Bước 3: Biên dịch mô hình
    model.compile(
        optimizer='adam',  # Tối ưu hóa bằng Adam (tự điều chỉnh learning rate)
        loss='categorical_crossentropy',  # Hàm mất mát cho phân loại đa lớp
        metrics=['accuracy']  # Theo dõi độ chính xác
    )
    
    # Bước 4: Huấn luyện mô hình
    history = model.fit(
        x_train, y_train,  # Dữ liệu huấn luyện
        epochs=10,  # Số lần duyệt qua toàn bộ dữ liệu
        batch_size=64,  # Số mẫu mỗi lần cập nhật trọng số
        validation_data=(x_test, y_test)  # Dữ liệu kiểm tra để đánh giá
    )
    # - Trong mỗi epoch:
    #   + Chia dữ liệu thành các batch (60000 / 64 ≈ 938 batch).
    #   + Tính gradient của hàm mất mát dựa trên batch.
    #   + Cập nhật trọng số bằng Adam.
    
    # Bước 5: Lưu mô hình
    os.makedirs('outputs/models', exist_ok=True)
    model.save('outputs/models/cnn_fashion_mnist.h5')
    print("Mô hình đã được lưu vào 'outputs/models/cnn_fashion_mnist.h5'")
    # Lưu history
    with open('outputs/models/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("History đã được lưu vào 'outputs/models/history.pkl'")
    
    return model, history

if __name__ == "__main__":
    model, history = train(use_saved_data=False)