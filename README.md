# Panorama Stitcher

## Giới thiệu

Panorama Stitcher là một ứng dụng Python để ghép nhiều ảnh thành một bức ảnh panorama liền mạch. Ứng dụng hỗ trợ cả giao diện đồ họa người dùng (GUI) sử dụng Tkinter và giao diện dòng lệnh (CLI). Nó sử dụng các kỹ thuật tiên tiến từ OpenCV như phát hiện đặc trưng SIFT, tính toán homography (RANSAC hoặc linear), và các phương pháp pha trộn (blending) như simple, feather, hoặc multiband. Người dùng có thể ghép ảnh từ file hoặc chụp trực tiếp từ camera.

### Tính năng chính:
- Giao diện GUI thân thiện để chọn ảnh, chụp từ camera, và xem trước kết quả.
- Hỗ trợ CLI để xử lý hàng loạt ảnh.
- Tự động cắt bỏ viền đen (auto-crop) để tạo panorama gọn gàng.
- Hỗ trợ các phương pháp pha trộn và homography khác nhau để tối ưu hóa chất lượng ảnh.

## Cài đặt

### Yêu cầu hệ thống
- Python 3.7 hoặc cao hơn.
- Hệ điều hành: Windows, Linux, hoặc macOS.
- Camera tương thích (cho chế độ chụp trực tiếp).
- (Tùy chọn) GStreamer trên Linux để cải thiện hiệu suất camera.

### Cài đặt thư viện

```bash
# Sao chép kho lưu trữ
git clone https://github.com/Kienk66-it/panorama-stitcher.git
cd panorama-stitcher

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# (Tùy chọn) Trên Linux, cài đặt GStreamer để hỗ trợ camera
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-good
````

## Sử dụng

### Chế độ GUI

Chạy ứng dụng với giao diện đồ họa:

```bash
python panorama_stitcher.py --gui
```

* Chế độ File: Chọn nhiều ảnh từ máy tính, cấu hình phương pháp pha trộn (blending) và homography, sau đó tạo panorama.
* Chế độ Camera: Kết nối camera, chụp ảnh, và ghép chúng thành panorama.

### Chế độ CLI

Ghép ảnh từ dòng lệnh:

```bash
python panorama_stitcher.py image1.jpg image2.jpg [output.jpg] [blend_method] [homography_method]
```

Ví dụ:

```bash
python panorama_stitcher.py img1.jpg img2.jpg panorama.jpg feather ransac
```

* Phương pháp pha trộn: `simple`, `feather`, `multiband`.
* Phương pháp homography: `linear`, `ransac`.
* Tùy chọn bổ sung:

  * `--no-crop`: Tắt tự động cắt viền.
  * `--display`: Hiển thị kết quả bằng Matplotlib.

## Lưu ý

* Đảm bảo máy tính có đủ bộ nhớ (khuyến nghị 8GB RAM) cho các panorama lớn.
* Tệp tạm được lưu trong `/tmp/panorama_captures` (Linux) hoặc thư mục tạm của hệ thống.
* Trên Windows, camera sử dụng backend OpenCV thay vì GStreamer.

<!--## License

Dự án này được cấp phép theo MIT License. Xem chi tiết trong tệp LICENSE. -->

## Liên hệ

Nếu bạn có câu hỏi hoặc cần hỗ trợ, hãy:

* Mở một issue trên GitHub.
* Gửi email tới: [lekien66.it@gmail.com](mailto:lekien66.it@gmail.com)
