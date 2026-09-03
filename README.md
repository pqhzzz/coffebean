# ☕ Coffee Roasting Classification using ResNet50

Bài tập thực hành môn **Thị giác Máy tính (Computer Vision)** - Xây dựng và triển khai mô hình học sâu phân loại cấp độ rang hạt cà phê.

---

## 🛠️ Công nghệ & Thư viện sử dụng
Dự án được xây dựng dựa trên các công nghệ và thư viện mã nguồn mở sau:
* **Python 3.13** - Ngôn ngữ lập trình chính.
* **PyTorch & Torchvision** - Xây dựng, huấn luyện và nạp trọng số mô hình mạng nơ-ron tích chập **ResNet50**.
* **Scikit-learn** - Tính toán và đánh giá hiệu năng mô hình (Confusion Matrix).
* **Matplotlib** - Trực quan hóa dữ liệu và biểu đồ.
* **Gradio** - Xây dựng giao diện web demo trực quan.
* **Pillow (PIL)** - Xử lý và đọc định dạng hình ảnh.

---

## 📁 Cấu trúc Thư mục Dự án
```text
CV_Coffee_Classification/
│
├── weights/
│   └── coffee_roasting_model.pth      # Tệp trọng số mô hình đã huấn luyện (ResNet50)
│
├── sample_images/                     # Thư mục chứa ảnh mẫu dùng để test nhanh
│   ├── dark_sample.jpg
│   └── light_sample.jpg
│
├── evaluate.py                        # Script tính toán và vẽ Ma trận nhầm lẫn (Confusion Matrix)
├── app_demo.py                        # Script chạy giao diện Web tương tác (Gradio)
├── requirements.txt                   # Danh sách các thư viện Python bắt buộc
└── README.md                          # Tài liệu hướng dẫn sử dụng dự án
