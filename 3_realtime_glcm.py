"""
FILE: 3_realtime_glcm.py
TUJUAN: Deteksi ekspresi wajah real-time menggunakan model
        Hybrid GLCM + MobileNetV2 + CNN.

CARA PAKAI:
    Pastikan sudah jalankan: python 2_train_glcm.py
    Lalu jalankan: python 3_realtime_glcm.py
    Tekan Q untuk keluar.

INSTALL:
    pip install scikit-image
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from torchvision import transforms, models
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "models/model_glcm_full.pth"
KAMERA_ID  = 1

with open("models/config.json", "r") as f:
    cfg = json.load(f)

IMG_SIZE = cfg["img_size"]   # 128
EKSPRESI = [e.capitalize() for e in cfg["ekspresi"]]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GLCM konfigurasi — HARUS sama persis dengan 2_train_glcm.py
GLCM_DISTANCES  = [1]
GLCM_ANGLES     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'energy', 'homogeneity', 'correlation']
GLCM_DIM        = len(GLCM_PROPERTIES) * len(GLCM_ANGLES)  # 16

# Post-processing correction
FAKTOR_KOREKSI = {
    'Angry'   : 15.5,
    'Disgust' : 0.0001,
    'Fear'    : 10.5,
    'Happy'   : 0.01,
    'Neutral' : 10.3,
    'Sad'     : 10.3,
    'Surprise': 10.3,
}

# Warna per ekspresi (BGR)
WARNA = {
    'Angry'   : (0,   0,   220),
    'Disgust' : (0,   128, 0),
    'Fear'    : (128, 0,   128),
    'Happy'   : (0,   200, 255),
    'Neutral' : (180, 180, 180),
    'Sad'     : (200, 80,  0),
    'Surprise': (0,   165, 255),
}

LABEL_ID = {
    'Angry':'MARAH', 'Disgust':'JIJIK', 'Fear':'TAKUT',
    'Happy':'SENANG', 'Neutral':'NETRAL',
    'Sad':'SEDIH', 'Surprise':'KAGET'
}


# ============================================================
# KELAS: Model (harus identik dengan 2_train_glcm.py)
# ============================================================
class HybridFERModelGLCM(nn.Module):
    def __init__(self, num_classes=7, glcm_dim=16):
        super(HybridFERModelGLCM, self).__init__()

        mobilenet = models.mobilenet_v2(pretrained=False)
        self.base_model = mobilenet.features

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.cnn_kustom = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.glcm_branch = nn.Sequential(
            nn.Linear(glcm_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, img, glcm):
        x_deep = self.base_model(img)
        x_deep = self.cnn_kustom(x_deep)
        x_deep = x_deep.view(x_deep.size(0), -1)
        x_glcm = self.glcm_branch(glcm)
        x_fusi = torch.cat([x_deep, x_glcm], dim=1)
        return self.classifier(x_fusi)


# ============================================================
# FUNGSI: Ekstraksi GLCM dari 1 gambar
# ============================================================
def ekstrak_glcm(img_pil):
    """
    Ekstrak 16 fitur GLCM dari gambar PIL.
    Identik dengan fungsi di 2_train_glcm.py.
    """
    img_gray  = np.array(img_pil.convert('L'))
    img_kuant = (img_gray / 4).astype(np.uint8)

    glcm = graycomatrix(
        img_kuant,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True
    )

    fitur = []
    for prop in GLCM_PROPERTIES:
        nilai = graycoprops(glcm, prop)[0]
        fitur.extend(nilai.tolist())

    return np.array(fitur, dtype=np.float32)


# ============================================================
# KELAS: Detektor Ekspresi Real-time dengan GLCM
# ============================================================
class DetektorEkspresiGLCM:

    def __init__(self):

        # ---- Load model ----
        print("Memuat model GLCM + MobileNetV2...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {MODEL_PATH}\n"
                "Jalankan dulu: python 2_train_glcm.py"
            )

        self.model = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
        self.model.to(DEVICE)
        self.model.eval()
        print(f"✅ Model dimuat ({DEVICE})")

        # ---- Load Haar Cascade ----
        cascade_path = (
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print("✅ Haar Cascade dimuat")

        # ---- Transformasi gambar ----
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # ---- Info fitur GLCM ----
        print(f"\nGLCM: {len(GLCM_PROPERTIES)} fitur × "
              f"{len(GLCM_ANGLES)} arah = {GLCM_DIM} dimensi")
        print(f"Fitur: {', '.join(GLCM_PROPERTIES)}")

        # ---- Faktor koreksi ----
        print("\nFaktor koreksi aktif:")
        for nama, f in FAKTOR_KOREKSI.items():
            status = "← dikurangi" if f < 1.0 else ("← dinaikkan" if f > 1.0 else "← normal")
            print(f"  {nama:<12} x{f:.1f}  {status}")

        self.waktu_sebelumnya = 0
        # Simpan nilai GLCM terakhir untuk ditampilkan
        self.glcm_nilai_terakhir = None

    # ----------------------------------------------------------
    # FUNGSI: Preprocessing gambar
    # ----------------------------------------------------------
    def preprocess_img(self, wajah_bgr):
        wajah_rgb = cv2.cvtColor(wajah_bgr, cv2.COLOR_BGR2RGB)
        wajah_pil = Image.fromarray(wajah_rgb)
        tensor    = self.transform(wajah_pil)
        return tensor.unsqueeze(0).to(DEVICE), wajah_pil

    # ----------------------------------------------------------
    # FUNGSI: Prediksi ekspresi + GLCM
    # ----------------------------------------------------------
    def prediksi(self, wajah_bgr):
        """
        Pipeline prediksi:
        1. Preprocess gambar → tensor
        2. Ekstrak fitur GLCM → tensor
        3. Model(img_tensor, glcm_tensor) → probabilitas
        4. Terapkan faktor koreksi
        5. Normalisasi ulang
        """

        # Preprocess gambar
        img_tensor, wajah_pil = self.preprocess_img(wajah_bgr)

        # Ekstrak GLCM
        glcm_fitur  = ekstrak_glcm(wajah_pil)
        glcm_tensor = torch.FloatTensor(glcm_fitur).unsqueeze(0).to(DEVICE)

        # Simpan nilai GLCM untuk ditampilkan di layar
        self.glcm_nilai_terakhir = glcm_fitur

        # Prediksi model
        with torch.no_grad():
            output = self.model(img_tensor, glcm_tensor)

        probs_asli = output.cpu().numpy()[0]

        # Post-processing koreksi
        faktor = np.array([FAKTOR_KOREKSI[n] for n in EKSPRESI])
        probs_koreksi = probs_asli * faktor
        total = probs_koreksi.sum()
        if total > 0:
            probs_koreksi = probs_koreksi / total

        idx_dominan    = np.argmax(probs_koreksi)
        nama_ekspresi  = EKSPRESI[idx_dominan]
        persen_dominan = probs_koreksi[idx_dominan] * 100

        return nama_ekspresi, persen_dominan, probs_koreksi

    # ----------------------------------------------------------
    # FUNGSI: Gambar anotasi wajah
    # ----------------------------------------------------------
    def gambar_anotasi(self, frame, x, y, w, h, nama, persen):
        warna = WARNA[nama]
        cv2.rectangle(frame, (x, y), (x+w, y+h), warna, 2)

        label = f"{nama}: {persen:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x, y-th-14), (x+tw+8, y), warna, -1)
        cv2.putText(frame, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        label_id = LABEL_ID.get(nama, nama)
        cv2.putText(frame, label_id, (x, y+h+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 2)

    # ----------------------------------------------------------
    # FUNGSI: Gambar bar probabilitas
    # ----------------------------------------------------------
    def gambar_bar(self, frame, probs, start_x, start_y):
        BAR_MAX = 130; BAR_H = 17; GAP = 4

        for i, (nama, prob) in enumerate(zip(EKSPRESI, probs)):
            y_bar  = start_y + i * (BAR_H + GAP)
            persen = prob * 100
            warna  = WARNA[nama]

            cv2.rectangle(frame,
                          (start_x+60, y_bar),
                          (start_x+60+BAR_MAX, y_bar+BAR_H),
                          (50,50,50), -1)
            isi = int(BAR_MAX * prob)
            if isi > 0:
                cv2.rectangle(frame,
                              (start_x+60, y_bar),
                              (start_x+60+isi, y_bar+BAR_H),
                              warna, -1)
            cv2.putText(frame, nama[:4],
                        (start_x, y_bar+BAR_H-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
            cv2.putText(frame, f"{persen:.0f}%",
                        (start_x+60+BAR_MAX+4, y_bar+BAR_H-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)

    # ----------------------------------------------------------
    # FUNGSI: Tampilkan nilai GLCM di layar
    # ----------------------------------------------------------
    def gambar_glcm_info(self, frame, nilai_glcm):
        """
        Tampilkan 4 nilai fitur GLCM rata-rata (dari 4 arah)
        di pojok kiri bawah layar.
        Berguna untuk demo ke dosen — menunjukkan GLCM aktif!
        """
        if nilai_glcm is None:
            return

        h = frame.shape[0]
        start_y = h - 110

        # Background semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, start_y-10),
                      (220, h-8), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "GLCM Features:",
                    (12, start_y+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,180), 1)

        # Rata-rata tiap fitur dari 4 arah
        n_arah = len(GLCM_ANGLES)
        for i, prop in enumerate(GLCM_PROPERTIES):
            # Nilai fitur ini: indeks i*n_arah sampai (i+1)*n_arah
            rata = np.mean(nilai_glcm[i*n_arah:(i+1)*n_arah])
            teks = f"{prop[:6]:<8}: {rata:.4f}"
            cv2.putText(frame, teks,
                        (12, start_y + 24 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (200,255,200), 1)

    # ----------------------------------------------------------
    # FUNGSI: Tampilkan FPS dan info
    # ----------------------------------------------------------
    def tampilkan_info(self, frame):
        sekarang = time.time()
        fps = 1.0 / max(sekarang - self.waktu_sebelumnya, 1e-9)
        self.waktu_sebelumnya = sekarang

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0,255,0), 2)
        cv2.putText(frame, "GLCM + MobileNetV2",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,200,255), 1)
        cv2.putText(frame, "Post-processing: ON",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,200,255), 1)

    # ----------------------------------------------------------
    # FUNGSI: Proses 1 frame
    # ----------------------------------------------------------
    def proses_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        wajah_list = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(48, 48)
        )

        if len(wajah_list) == 0:
            cv2.putText(frame, "Tidak ada wajah terdeteksi",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,165,255), 2)
        else:
            for (x, y, w, h) in wajah_list:
                margin = 10
                x1 = max(0, x-margin); y1 = max(0, y-margin)
                x2 = min(frame.shape[1], x+w+margin)
                y2 = min(frame.shape[0], y+h+margin)
                wajah_crop = frame[y1:y2, x1:x2]

                if wajah_crop.size == 0:
                    continue

                nama, persen, probs = self.prediksi(wajah_crop)
                self.gambar_anotasi(frame, x, y, w, h, nama, persen)

                bar_x = min(x+w+15, frame.shape[1]-210)
                bar_y = max(y, 10)
                self.gambar_bar(frame, probs, bar_x, bar_y)

        # Tampilkan nilai GLCM di pojok bawah
        self.gambar_glcm_info(frame, self.glcm_nilai_terakhir)

        self.tampilkan_info(frame)

        cv2.putText(frame, "Tekan Q untuk keluar",
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180,180,180), 1)

        return frame

    # ----------------------------------------------------------
    # FUNGSI: Loop utama webcam
    # ----------------------------------------------------------
    def jalankan(self):
        print(f"\nMembuka webcam (ID: {KAMERA_ID})...")
        cap = cv2.VideoCapture(KAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError(
                "Tidak bisa membuka webcam!\n"
                f"Coba ganti KAMERA_ID = {KAMERA_ID} ke 1 atau 2"
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        print("✅ Webcam aktif! Tekan Q untuk keluar.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame.")
                break

            frame = cv2.flip(frame, 1)
            frame = self.proses_frame(frame)
            cv2.imshow(
                'Deteksi Ekspresi — GLCM + MobileNetV2 + CNN',
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nKeluar.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Program selesai.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*55)
    print("  DETEKSI EKSPRESI REAL-TIME")
    print("  Hybrid GLCM + MobileNetV2 + CNN")
    print("="*55)

    # Install check
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        print("❌ scikit-image belum terinstall!")
        print("   Jalankan: pip install scikit-image")
        exit(1)

    detektor = DetektorEkspresiGLCM()
    detektor.jalankan()
