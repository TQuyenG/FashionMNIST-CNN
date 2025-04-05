"""
File: data.py
Mục đích: Dùng để tải, tiền xử lý và lưu dữ liệu Fashion-MNIST.
- Cho file nào/cái gì? Cung cấp dữ liệu đã xử lý cho `train.py` và `evaluate.py`, 
  đồng thời lưu dữ liệu thành file .npy (cho máy học) và .csv (cho người dùng).

Quy trình:
1. Tải dữ liệu thô từ Fashion-MNIST qua TensorFlow.
2. Tiền xử lý:
   - Chuẩn hóa giá trị pixel về [0, 1].
   - Reshape ảnh từ (28, 28) thành (28, 28, 1) cho CNN.
   - Chuyển nhãn thành one-hot encoding.
3. (Tùy chọn) Lưu dữ liệu:
   - Lưu .npy: Dữ liệu ảnh và nhãn dạng mảng NumPy.
   - Lưu .csv: Dữ liệu ảnh phẳng (784 cột) và nhãn gốc.
4. Trả về dữ liệu đã xử lý cho các file khác.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os

def load_and_preprocess_data(save_files=False):
    # Bước 1: Tải dữ liệu thô
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Bước 2: Tiền xử lý
    # Chuẩn hóa pixel về [0, 1] để mô hình học nhanh hơn
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape ảnh để thêm chiều kênh (channel) cho CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Chuyển nhãn thành one-hot encoding cho phân loại đa lớp
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
    
    # Bước 3: Lưu dữ liệu (nếu save_files=True)
    if save_files:
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/test', exist_ok=True)
        
        # Lưu file .npy cho máy học
        np.save('data/train/x_train.npy', x_train)
        np.save('data/train/y_train.npy', y_train_onehot)
        np.save('data/test/x_test.npy', x_test)
        np.save('data/test/y_test.npy', y_test_onehot)
        
        # Flatten ảnh để lưu .csv
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        
        # Lưu file .csv cho người dùng
        train_df = pd.DataFrame(x_train_flat, columns=[f"pixel_{i}" for i in range(784)])
        train_df['label'] = y_train
        train_df.to_csv('data/train/train_data.csv', index=False)
        
        test_df = pd.DataFrame(x_test_flat, columns=[f"pixel_{i}" for i in range(784)])
        test_df['label'] = y_test
        test_df.to_csv('data/test/test_data.csv', index=False)
    
    # Bước 4: Trả về dữ liệu đã xử lý
    return (x_train, y_train_onehot), (x_test, y_test_onehot)

def load_from_files():
    # Tải dữ liệu từ file .npy nếu đã lưu trước đó
    x_train = np.load('data/train/x_train.npy')
    y_train = np.load('data/train/y_train.npy')
    x_test = np.load('data/test/x_test.npy')
    y_test = np.load('data/test/y_test.npy')
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(save_files=True)
    print("Dữ liệu đã được lưu vào 'data/train/' và 'data/test/'")