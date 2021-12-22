Hướng dẫn các bước chạy chương trình: 

Bước 1: Tải phần mềm visual studio, và môi trường lập trình với python.

Bước 2: Mở folder trong Visual Studio: Chọn file->Open Folfer-> Chọn thư mục face_recognition vừa tải về. 

# Nếu đã có môi trường
Bước 3: Kết nối với môi trường python trên Visual Code: nhập tổ hợp phím Ctrl+Shift+P, chọn Select Interpreter. Vào View-> Terminal, mở của sổ ternimal, nhấn chuột dấu cộng bên phải góc trên của cửa sổ Terminal, chọn Command Prompt.

# Nếu chưa có môi trường
Bước 3: Trong của sổ Terminal nhập "py -3 -m venv avenv". Sau đó kích hoạt môi trường bằng lệnh "avenv\scripts\activate"

Bước 4: Tải các thư viện cần thiết cho chương trình: nhập vào cửa sổ Terminal “pip install opencv-python matplotlib tensorflow keras pillow”

Bước 5: Chạy chương trình nhận diện: nhập vào cửa sổ Terminal “python UI_recognition.py”, sau đó chọn một tấm ảnh trong thư mục Test Image rồi nhấn face-recognition để hệ thống tiến hành nhận diện khuôn mặt.# DOAN-II