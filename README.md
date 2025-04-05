# FashionMNIST CNN

- Dự án này sử dụng Convolutional Neural Network (CNN) để phân loại 10 loại quần áo từ tập dữ liệu Fashion-MNIST.  
- Fashion-MNIST bao gồm 70,000 ảnh grayscale (28x28 pixel), với 60,000 ảnh huấn luyện và 10,000 ảnh kiểm tra, chia thành 10 lớp: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Mục tiêu
- Xây dựng mô hình CNN để phân loại chính xác các loại quần áo.
- Đánh giá hiệu suất mô hình qua các biểu đồ (confusion matrix, accuracy, loss).
- Dự đoán trên tập kiểm tra và ảnh tùy chọn từ người dùng, hiển thị xác suất từng lớp.

## Cài đặt

### Tạo môi trường ảo

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Cách chạy

### Chuẩn bị dữ liệu

```bash
python src/data.py
```

### Huấn luyện mô hình

```bash
python -m src.train
```

### Đánh giá mô hình

```bash
python -m src.evaluate
```
