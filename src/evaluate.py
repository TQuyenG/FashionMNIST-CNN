"""
File: evaluate.py
Mục đích: Dùng để đánh giá mô hình đã huấn luyện và dự đoán kết quả.
- Cho file nào/cái gì? 
  - Hàm evaluate(): Tạo biểu đồ (confusion matrix, accuracy, loss) để đánh giá hiệu suất mô hình.
  - Hàm predict(): Dự đoán trên tập kiểm tra (và lưu .csv) cũng như trên ảnh tùy chọn từ người dùng, hiển thị tỉ lệ dự đoán.

Quy trình tổng thể:
1. Tải dữ liệu kiểm tra từ file .npy.
2. Tải mô hình đã huấn luyện từ file .h5 và history từ .pkl.
3. Đánh giá (evaluate):
   - Tính độ chính xác và mất mát trên tập kiểm tra.
   - Trực quan hóa: confusion matrix, accuracy/loss qua epoch (không dự đoán ở đây).
4. Dự đoán (predict):
   - Dự đoán nhãn và xác suất cho tập kiểm tra, hiển thị 20 mẫu ngẫu nhiên, lưu kết quả vào .csv.
   - Dự đoán trên ảnh tùy chọn: tiền xử lý, dự đoán, hiển thị xác suất từng lớp.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from src.data import load_from_files
import tensorflow as tf
import pandas as pd
import cv2
import pickle

# Danh sách tên lớp (dùng chung cho cả hai hàm)
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def evaluate():
    # [Giữ nguyên như trước, không thay đổi]
    (_, _), (x_test, y_test) = load_from_files()
    model = tf.keras.models.load_model('outputs/models/cnn_fashion_mnist.h5')
    with open('outputs/models/history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Độ chính xác trên tập kiểm tra: {test_acc:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/accuracy_plot.png')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss qua các epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/figures/loss_plot.png')
    plt.show()

def predict(image_path=None):
    # Bước 1: Tải dữ liệu kiểm tra
    (_, _), (x_test, y_test) = load_from_files()
    
    # Bước 2: Tải mô hình
    model = tf.keras.models.load_model('outputs/models/cnn_fashion_mnist.h5')
    
    # Bước 3: Dự đoán trên tập kiểm tra
    y_pred = model.predict(x_test)  # Dự đoán xác suất cho 10,000 ảnh
    y_pred_classes = np.argmax(y_pred, axis=1)  # Chuyển xác suất thành nhãn (0-9)
    y_test_classes = np.argmax(y_test, axis=1)  # Nhãn thật từ one-hot
    
    # Confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/confusion_matrix.png')
    plt.show()
    
    # Chọn 20 ảnh ngẫu nhiên từ tập kiểm tra
    indices = np.random.choice(len(x_test), 20, replace=False)  # Chọn 20 chỉ số ngẫu nhiên
    plt.figure(figsize=(20, 12))  # Tăng chiều cao figure để chứa 20 ảnh và text
    for i, idx in enumerate(indices):  # Lặp qua các chỉ số ngẫu nhiên
        plt.subplot(4, 5, i+1)  # Sắp xếp 20 ảnh thành lưới 4x5
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        pred_class = y_pred_classes[idx]
        true_class = y_test_classes[idx]
        pred_prob = y_pred[idx][pred_class] * 100  # Xác suất cao nhất (%)
        is_correct = "Đúng" if pred_class == true_class else "Sai"
        # Thêm chỉ số ảnh (index) vào tiêu đề
        plt.title(f"Ảnh {idx}\nPred: {class_names[pred_class]} ({pred_prob:.1f}%)\nTrue: {class_names[true_class]}\n{is_correct}",
                  fontsize=10, pad=5)  # Tăng khoảng cách text với ảnh (pad)
        plt.axis('off')
    plt.subplots_adjust(hspace=1)  # Tăng khoảng cách giữa các hàng (trên/dưới)
    plt.savefig('outputs/figures/sample_predictions.png')
    plt.show()
    
    # Lưu kết quả dự đoán vào .csv
    os.makedirs('outputs/predictions', exist_ok=True)
    pred_df = pd.DataFrame({
        'True_Label': y_test_classes,
        'Predicted_Label': y_pred_classes
    })
    pred_df.to_csv('outputs/predictions/predictions.csv', index=False)
    print("Kết quả dự đoán đã được lưu vào 'outputs/predictions/predictions.csv'")
    
    # Bước 4: Dự đoán trên ảnh tùy chọn (nếu có)
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Không thể đọc ảnh từ {image_path}")
            return
        
        img_resized = cv2.resize(img, (28, 28))
        img_normalized = img_resized.astype('float32') / 255.0
        img_input = img_normalized.reshape(1, 28, 28, 1)
        
        pred = model.predict(img_input)[0]
        pred_class = np.argmax(pred)
        pred_label = class_names[pred_class]
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_resized, cmap='gray')
        plt.title(f"Dự đoán: {pred_label} ({pred[pred_class]*100:.1f}%)")
        plt.axis('off')
        plt.show()
        
        print(f"\nDự đoán cho ảnh {image_path}:")
        for i, prob in enumerate(pred):
            print(f"- {class_names[i]}: {prob*100:.2f}%")
        print(f"Kết quả: {pred_label} (xác suất cao nhất: {pred[pred_class]*100:.2f}%)")

if __name__ == "__main__":
    # evaluate()
    predict(image_path="D:/test1.png")