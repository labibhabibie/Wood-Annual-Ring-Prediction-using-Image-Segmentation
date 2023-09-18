import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fungsi untuk menghitung umur pohon berdasarkan cincin tahunan kayu
def hitung_umur_pohon(citra):
    # Konversi citra ke mode warna Grayscale
    grayscale_citra = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)

    # Terapkan teknik thresholding untuk mengidentifikasi cincin tahunan kayu
    _, binary_citra = cv2.threshold(grayscale_citra, 100, 255, cv2.THRESH_BINARY)

    # Terapkan K-Means clustering untuk mengelompokkan cincin tahunan kayu
    pixels = np.float32(citra.reshape(-1, 3))
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)

    # Dapatkan label K-Means
    labels = kmeans.labels_

    # Hitung jumlah cincin tahunan kayu berdasarkan label K-Means
    umur_pohon = len(np.unique(labels))

    return umur_pohon

# Baca citra pohon
citra_pohon = cv2.imread('Data\images\PSM_V03_D334_Annual_ring_growth.jpg')  # Ganti 'citra_pohon.jpg' dengan nama citra Anda
citra_pohon = cv2.cvtColor(citra_pohon, cv2.COLOR_BGR2RGB)

# Hitung umur pohon
umur_pohon = hitung_umur_pohon(citra_pohon)

# Tampilkan hasil
plt.figure(figsize=(8, 8))
plt.imshow(citra_pohon, cmap='gray')
plt.axis('off')
plt.title(f'Umur Pohon: {umur_pohon} tahun')
plt.show()
