# trainning_model.py
import numpy as np
import scipy.signal as signal
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def plot_svm_decision_boundary(model, X, y):
    # ลดมิติข้อมูลเป็น 2 มิติด้วย PCA เพื่อให้สามารถพล็อตกราฟได้
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # สร้าง mesh grid สำหรับ plotting boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # คำนวณ prediction บน grid
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # พล็อตข้อมูลการแบ่งเขตการตัดสินใจ
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='winter')
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='winter', edgecolors='k')
    
    # พล็อต Support Vectors
    support_vectors = model.support_vectors_
    support_vectors_pca = pca.transform(support_vectors)
    plt.scatter(support_vectors_pca[:, 0], support_vectors_pca[:, 1], s=100,
                facecolors='none', edgecolors='r', label='Support Vectors')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundary with Support Vectors')
    plt.legend()
    fig = plt.gcf()
    plt.close(fig)
    return fig

def extract_features(y, sr, low_cutoff=500, high_cutoff=2500):
    # กรองช่วงความถี่ด้วย band-pass filter
    sos = signal.butter(10, [low_cutoff, high_cutoff], btype='bandpass', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos, y)
    
    # คำนวณ MFCC 40 ค่า
    mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=40)
    mfcc_means = np.mean(mfccs, axis=1)
    
    # คำนวณ Spectral Centroid และ Bandwidth
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_filtered, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr))
    
    # คำนวณ Zero Crossing Rate
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y_filtered))
    
    features = np.concatenate((mfcc_means, [spectral_centroid, spectral_bandwidth, zero_crossing_rate]))
    return features, y_filtered

def load_data(good_dir, bad_dir, low_cutoff=500, high_cutoff=2500):
    features = []
    labels = []
    # โหลดไฟล์เสียงจากโฟลเดอร์ good
    for file in os.listdir(good_dir):
        file_path = os.path.join(good_dir, file)
        y, sr = librosa.load(file_path, sr=None)
        feature, _ = extract_features(y, sr, low_cutoff, high_cutoff)
        features.append(feature)
        labels.append(0)  # 0 แทน good material

    # โหลดไฟล์เสียงจากโฟลเดอร์ bad
    for file in os.listdir(bad_dir):
        file_path = os.path.join(bad_dir, file)
        y, sr = librosa.load(file_path, sr=None)
        feature, _ = extract_features(y, sr, low_cutoff, high_cutoff)
        features.append(feature)
        labels.append(1)  # 1 แทน faulty material

    # สมมติว่า sample rate ของไฟล์ทั้งหมดเท่ากัน
    return np.array(features), np.array(labels), sr

def save_to_excel(features, labels, sr, low_cutoff, high_cutoff, output_file="features_data.xlsx"):
    mfcc_columns = [f"MFCC_{i+1}" for i in range(40)]
    spectral_columns = ["Spectral_Centroid", "Spectral_Bandwidth", "Zero_Crossing_Rate"]
    
    data = pd.DataFrame(features, columns=mfcc_columns + spectral_columns)
    data["Label"] = labels
    
    metadata = {
        "Sampling_Rate": [sr],
        "Low_Cutoff_Frequency": [low_cutoff],
        "High_Cutoff_Frequency": [high_cutoff]
    }
    metadata_df = pd.DataFrame(metadata)
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Feature_Data", index=False)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
    print(f"Data saved to {output_file}")

def train_model(good_dir, bad_dir, low_cutoff=500, high_cutoff=2500):
    """
    ฟังก์ชัน train_model ทำหน้าที่โหลดข้อมูลจากโฟลเดอร์ good และ bad,
    สกัดฟีเจอร์, แบ่งข้อมูลเป็น train/test, ฝึกโมเดล SVM,
    บันทึกโมเดลและ scaler และคืนค่าผลลัพธ์
    """
    # โหลดข้อมูลและเตรียมฟีเจอร์
    features, labels, sr = load_data(good_dir, bad_dir, low_cutoff, high_cutoff)
    
    # แบ่งข้อมูลเป็น train และ test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=50)
    
    # ปรับขนาดข้อมูลด้วย StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ฝึกโมเดล SVM
    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)
    
    # ทดสอบโมเดลและคำนวณความแม่นยำ
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    
    # บันทึกโมเดลและ scaler
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler, accuracy, features, labels, sr

if __name__ == "__main__":
    # โค้ดสำหรับทดสอบเมื่อรันไฟล์นี้โดยตรง
    good_dir = r'C:\Users\SPARK\OneDrive\Desktop\project\อิฐ\GG\1'
    bad_dir = r'C:\Users\SPARK\OneDrive\Desktop\project\อิฐ\FL\1'
    low_cutoff = 500
    high_cutoff = 2500
    model, scaler, accuracy, features, labels, sr = train_model(good_dir, bad_dir, low_cutoff, high_cutoff)
    fig = plot_svm_decision_boundary(model, features, labels)
    fig.show()
    save_to_excel(features, labels, sr, low_cutoff, high_cutoff, output_file="features_data.xlsx")
