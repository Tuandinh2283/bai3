import cv2
import numpy as np
import time
from sklearn import svm, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Bước 1: Tiền xử lý ảnh và Trích xuất đặc trưng
def extract_color_histogram(image, bins=(8, 8, 8)):
    """Trích xuất biểu đồ màu 3D từ ảnh và chuẩn hóa."""
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_and_preprocess_image(image_path):
    """Tải ảnh từ đường dẫn và trích xuất đặc trưng."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Thay đổi kích thước ảnh để nhất quán
    features = extract_color_histogram(image)
    return features

# Bước 2: Mô phỏng tập dữ liệu (cho mục đích minh họa, tạo một tập dữ liệu ngẫu nhiên)
def simulate_dataset(n_samples=100, n_classes=2):
    """Mô phỏng một tập dữ liệu để minh họa với dữ liệu ngẫu nhiên."""
    np.random.seed(42)  # Đảm bảo tái tạo kết quả
    X = np.random.rand(n_samples, 512)  # Các vector đặc trưng giả lập
    y = np.random.randint(0, n_classes, n_samples)  # Nhãn giả lập
    return X, y

# Bước 3: Huấn luyện các mô hình
def train_and_evaluate_models(X, y):
    """Huấn luyện các mô hình SVM, KNN, và Decision Tree và đánh giá chúng."""
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Các mô hình cần huấn luyện
    models = {
        "SVM": svm.SVC(kernel='linear'),
        "KNN": neighbors.KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": tree.DecisionTreeClassifier()
    }

    results = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Dự đoán
        y_pred = model.predict(X_test)

        # Tính toán các chỉ số
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Time": train_time
        }
    return results

# Bước 4: Tổng hợp lại
def main(image_path):
    # Tải và tiền xử lý ảnh (ví dụ về trích xuất đặc trưng cho một ảnh)
    image_features = load_and_preprocess_image(image_path)

    # Mô phỏng một tập dữ liệu và thêm đặc trưng ảnh của chúng ta
    X, y = simulate_dataset()
    X = np.vstack([X, image_features])  # Thêm đặc trưng ảnh vào tập dữ liệu
    y = np.append(y, 1)  # Giả định ảnh của chúng ta thuộc lớp '1'

    # Huấn luyện và đánh giá các mô hình
    results = train_and_evaluate_models(X, y)

    # Hiển thị kết quả
    for model_name, metrics in results.items():
        print(f"Kết quả cho {model_name}:")
        print(f"  Độ chính xác (Accuracy): {metrics['Accuracy']:.4f}")
        print(f"  Độ chính xác trung bình (Precision): {metrics['Precision']:.4f}")
        print(f"  Độ hồi tưởng (Recall): {metrics['Recall']:.4f}")
        print(f"  Thời gian huấn luyện (Training Time): {metrics['Time']:.4f} giây\n")


# Chạy hàm chính với ảnh đã tải lên
main('Chumeodethuong_Wap102 (1).jpg')