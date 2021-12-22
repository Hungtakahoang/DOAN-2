import cv2
import os
import shutil
import numpy as np
# xóa thư mục cũ
folderDIR_1 = "../face_recognition/Train_data_new"
folderDIR_2 = "../face_recognition/Valid_data_new"

folder_train1 = r'D:\opencv-master\face_recognition\Train_data_new\Albert Einstein'
folder_train2 = r'D:\opencv-master\face_recognition\Train_data_new\Joe Biden'
folder_train3 = r'D:\opencv-master\face_recognition\Train_data_new\Cristiano Ronaldo'
folder_train4 = r'D:\opencv-master\face_recognition\Train_data_new\Donald Trump'
folder_train5 = r'D:\opencv-master\face_recognition\Train_data_new\Galileo Galilei'
folder_train6 = r'D:\opencv-master\face_recognition\Train_data_new\Lionel Messi'
folder_train7 = r'D:\opencv-master\face_recognition\Train_data_new\Mark Zuckerberg'
folder_train8 = r'D:\opencv-master\face_recognition\Train_data_new\Nicola Tesla'
folder_train9 = r'D:\opencv-master\face_recognition\Train_data_new\Son Tung MTP'
folder_train10 = r'D:\opencv-master\face_recognition\Train_data_new\Tran Thanh'

folder_valid1 = r'D:\opencv-master\face_recognition\Valid_data_new\Albert Einstein'
folder_valid2 = r'D:\opencv-master\face_recognition\Valid_data_new\Joe Biden'
folder_valid3 = r'D:\opencv-master\face_recognition\Valid_data_new\Cristiano Ronaldo'
folder_valid4 = r'D:\opencv-master\face_recognition\Valid_data_new\Donald Trump'
folder_valid5 = r'D:\opencv-master\face_recognition\Valid_data_new\Galileo Galilei'
folder_valid6 = r'D:\opencv-master\face_recognition\Valid_data_new\Lionel Messi'
folder_valid7 = r'D:\opencv-master\face_recognition\Valid_data_new\Mark Zuckerberg'
folder_valid8 = r'D:\opencv-master\face_recognition\Valid_data_new\Nicola Tesla'
folder_valid9 = r'D:\opencv-master\face_recognition\Valid_data_new\Son Tung MTP'
folder_valid10 = r'D:\opencv-master\face_recognition\Valid_data_new\Tran Thanh'

# if os.path.exists(folderDIR_1):
#     shutil.rmtree(folderDIR_1)
# if os.path.exists(folderDIR_2):
#     shutil.rmtree(folderDIR_2)

# tạo thư mục dữ liệu mới, rỗng
if not os.path.exists(folderDIR_1):
    os.makedirs("Train_data_new", exist_ok=True)
if not os.path.exists(folderDIR_2):
    os.makedirs("Valid_data_new", exist_ok=True)

if not os.path.exists(folder_train1):
    os.makedirs("Train_data_new/Albert Einstein", exist_ok=True)
if not os.path.exists(folder_train2):
    os.makedirs("Train_data_new/Joe Biden", exist_ok=True)
if not os.path.exists(folder_train3):
    os.makedirs("Train_data_new/Cristiano Ronaldo", exist_ok=True)
if not os.path.exists(folder_train4):
    os.makedirs("Train_data_new/Donald Trump", exist_ok=True)
if not os.path.exists(folder_train5):
    os.makedirs("Train_data_new/Galileo Galilei", exist_ok=True)
if not os.path.exists(folder_train6):
    os.makedirs("Train_data_new/Lionel Messi", exist_ok=True)
if not os.path.exists(folder_train7):
    os.makedirs("Train_data_new/Mark Zuckerberg", exist_ok=True)
if not os.path.exists(folder_train8):
    os.makedirs("Train_data_new/Nicola Tesla", exist_ok=True)
if not os.path.exists(folder_train9):
    os.makedirs("Train_data_new/Son Tung MTP", exist_ok=True)
if not os.path.exists(folder_train10):
    os.makedirs("Train_data_new/Tran Thanh", exist_ok=True)

if not os.path.exists(folder_valid1):
    os.makedirs("Valid_data_new/Albert Einstein", exist_ok=True)
if not os.path.exists(folder_valid2):
    os.makedirs("Valid_data_new/Joe Biden", exist_ok=True)
if not os.path.exists(folder_valid3):
    os.makedirs("Valid_data_new/Cristiano Ronaldo", exist_ok=True)
if not os.path.exists(folder_valid4):
    os.makedirs("Valid_data_new/Donald Trump", exist_ok=True)
if not os.path.exists(folder_valid5):
    os.makedirs("Valid_data_new/Galileo Galilei", exist_ok=True)
if not os.path.exists(folder_valid6):
    os.makedirs("Valid_data_new/Lionel Messi", exist_ok=True)
if not os.path.exists(folder_valid7):
    os.makedirs("Valid_data_new/Mark Zuckerberg", exist_ok=True)
if not os.path.exists(folder_valid8):
    os.makedirs("Valid_data_new/Nicola Tesla", exist_ok=True)
if not os.path.exists(folder_valid9):
    os.makedirs("Valid_data_new/Son Tung MTP", exist_ok=True)
if not os.path.exists(folder_valid10):
    os.makedirs("Valid_data_new/Tran Thanh", exist_ok=True)

# hàm cắt khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        # printcv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faces = img[y:y + h, x:x + w]
    return faces

def image_processed(img_crop):
    faces_resize = cv2.resize(img_crop, (224, 224))
    # lọc nhiễu
    faces_blur = cv2.medianBlur(faces_resize, 3)
    # làm rõ ảnh
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    faces = cv2.filter2D(faces_blur, -1, kernel = kernel)
    return faces

# crop face for each folders
path_train1 = r'D:\opencv-master\face_recognition\datasets\train_data\Albert Einstein'
path_train2 = r'D:\opencv-master\face_recognition\datasets\train_data\Joe Biden'
path_train3 = r'D:\opencv-master\face_recognition\datasets\train_data\Cristiano Ronaldo'
path_train4 = r'D:\opencv-master\face_recognition\datasets\train_data\Donald Trump'
path_train5 = r'D:\opencv-master\face_recognition\datasets\train_data\Galileo Galilei'
path_train6 = r'D:\opencv-master\face_recognition\datasets\train_data\Lionel Messi'
path_train7 = r'D:\opencv-master\face_recognition\datasets\train_data\Mark Zuckerberg'
path_train8 = r'D:\opencv-master\face_recognition\datasets\train_data\Nicola Tesla'
path_train9 = r'D:\opencv-master\face_recognition\datasets\train_data\Son Tung MTP'
path_train10 = r'D:\opencv-master\face_recognition\datasets\train_data\Tran Thanh'

path_valid1 = r'D:\opencv-master\face_recognition\datasets\valid_data\Albert Einstein'
path_valid2 = r'D:\opencv-master\face_recognition\datasets\valid_data\Joe Biden'
path_valid3 = r'D:\opencv-master\face_recognition\datasets\valid_data\Cristiano Ronaldo'
path_valid4 = r'D:\opencv-master\face_recognition\datasets\valid_data\Donald Trump'
path_valid5 = r'D:\opencv-master\face_recognition\datasets\valid_data\Galileo Galilei'
path_valid6 = r'D:\opencv-master\face_recognition\datasets\valid_data\Lionel Messi'
path_valid7 = r'D:\opencv-master\face_recognition\datasets\valid_data\Mark Zuckerberg'
path_valid8 = r'D:\opencv-master\face_recognition\datasets\valid_data\Nicola Tesla'
path_valid9 = r'D:\opencv-master\face_recognition\datasets\valid_data\Son Tung MTP'
path_valid10 = r'D:\opencv-master\face_recognition\datasets\valid_data\Tran Thanh'

# numbers = len(os.listdir("../face_recognition/datasets/train_data"))
# print(numbers)

# cho chạy từng vòng lặp một
images = os.listdir(path_train1)
for image in images:
    img = cv2.imread(os.path.join(path_train1, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train1)
    cv2.imwrite(os.path.join(folder_train1, image), faces)

images = os.listdir(path_train2)
for image in images:
    img = cv2.imread(os.path.join(path_train2, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train2)
    cv2.imwrite(image, faces)

images = os.listdir(path_train3)
for image in images:
    img = cv2.imread(os.path.join(path_train3, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train3)
    cv2.imwrite(image, faces)

images = os.listdir(path_train4)
for image in images:
    img = cv2.imread(os.path.join(path_train4, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train4)
    cv2.imwrite(image, faces)

images = os.listdir(path_train5)
for image in images:
    img = cv2.imread(os.path.join(path_train5, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train5)
    cv2.imwrite(image, faces)

images = os.listdir(path_train6)
for image in images:
    img = cv2.imread(os.path.join(path_train6, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train6)
    cv2.imwrite(image, faces)

images = os.listdir(path_train7)
for image in images:
    img = cv2.imread(os.path.join(path_train7, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train7)
    cv2.imwrite(image, faces)
    
images = os.listdir(path_train8)
for image in images:
    img = cv2.imread(os.path.join(path_train8, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train8)
    cv2.imwrite(image, faces)

images = os.listdir(path_train9)
for image in images:
    img = cv2.imread(os.path.join(path_train9, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train9)
    cv2.imwrite(image, faces)

images = os.listdir(path_train10)
for image in images:
    img = cv2.imread(os.path.join(path_train10, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_train10)
    cv2.imwrite(image, faces)

# # ///////////////////////////////////
images = os.listdir(path_valid1)
for image in images:
    img = cv2.imread(os.path.join(path_valid1, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid1)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid2)
for image in images:
    img = cv2.imread(os.path.join(path_valid2, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid2)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid3)
for image in images:
    img = cv2.imread(os.path.join(path_valid3, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid3)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid4)
for image in images:
    img = cv2.imread(os.path.join(path_valid4, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid4)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid5)
for image in images:
    img = cv2.imread(os.path.join(path_valid5, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid5)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid6)
for image in images:
    img = cv2.imread(os.path.join(path_valid6, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid6)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid7)
for image in images:
    img = cv2.imread(os.path.join(path_valid7, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid7)
    cv2.imwrite(image, faces)
    
images = os.listdir(path_valid8)
for image in images:
    img = cv2.imread(os.path.join(path_valid8, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid8)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid9)
for image in images:
    img = cv2.imread(os.path.join(path_valid9, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid9)
    cv2.imwrite(image, faces)

images = os.listdir(path_valid10)
for image in images:
    img = cv2.imread(os.path.join(path_valid10, image))
    faces_crop = crop_face(img)
    faces = image_processed(faces_crop)
    os.chdir(folder_valid10)
    cv2.imwrite(image, faces)