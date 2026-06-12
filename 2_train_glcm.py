"""
FILE: 2_train.py — v4 GLCM Feature Fusion
TUJUAN: Training model hybrid MobileNetV2 + CNN + GLCM

PERUBAHAN dari v3:
- Tambah cabang GLCM (Gray Level Co-occurrence Matrix)
- GLCM mengekstrak 4 fitur × 4 arah = 16 dimensi fitur tekstur
- Fitur GLCM di-concatenate dengan fitur deep MobileNetV2 (128 dim)
- Total fitur sebelum classifier: 128 + 16 = 144 dimensi
- Classifier baru menerima 144 fitur → Dense(128) → Softmax(7)

INSTALL TAMBAHAN:
    pip install scikit-image

CARA PAKAI:
    Pastikan sudah jalankan: python 1_preprocessing.py
    Lalu jalankan: python 2_train.py
"""

import os, json, time, copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# ============================================================
# KONFIGURASI
# ============================================================
with open("models/config.json","r") as f:
    cfg = json.load(f)

DATASET_DIR = cfg["dataset_dir"]
IMG_SIZE    = cfg["img_size"]      # 128
BATCH_SIZE  = cfg["batch_size"]    # 64
RANDOM_SEED = cfg["random_seed"]
EKSPRESI    = cfg["ekspresi"]
NUM_CLASSES = len(EKSPRESI)        # 7

EPOCH_FASE1 = 20
EPOCH_FASE2 = 15

# GLCM: 4 fitur × 4 arah = 16 dimensi
GLCM_DISTANCES  = [1]              # jarak antar piksel
GLCM_ANGLES     = [0, np.pi/4,    # 4 arah: 0°, 45°, 90°, 135°
                   np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'energy', 'homogeneity', 'correlation']
GLCM_DIM        = len(GLCM_PROPERTIES) * len(GLCM_ANGLES)  # 4×4 = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# FUNGSI: Ekstraksi fitur GLCM dari 1 gambar
# ============================================================
def ekstrak_glcm(img_pil):
    """
    Ekstrak 16 fitur GLCM dari gambar PIL.

    Langkah:
    1. Konversi RGB → Grayscale (GLCM butuh 1 channel)
    2. Kuantisasi ke 64 level (dari 256) → matriks lebih stabil
    3. Hitung GLCM untuk 4 arah (0°, 45°, 90°, 135°)
    4. Ekstrak 4 properti: contrast, energy, homogeneity, correlation
    5. Hasilnya: 4 properti × 4 arah = 16 nilai

    Kenapa 4 arah?
    Ekspresi wajah punya pola tekstur berbeda di tiap arah.
    Dengan 4 arah, kita menangkap lebih banyak informasi tekstur.
    """

    # Konversi ke grayscale
    img_gray = np.array(img_pil.convert('L'))

    # Kuantisasi ke 64 level (0-255 → 0-63)
    # Mengurangi ukuran GLCM, membuat perhitungan lebih cepat
    img_kuant = (img_gray / 4).astype(np.uint8)

    # Hitung GLCM
    # distances=[1] → perhatikan piksel yang bertetangga langsung
    # angles    → 4 arah berbeda
    # levels=64 → sesuai kuantisasi di atas
    # symmetric=True → GLCM simetris (P[i,j] = P[j,i])
    # normed=True    → normalisasi sehingga total = 1
    glcm = graycomatrix(
        img_kuant,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True
    )

    # Ekstrak properti dari GLCM
    # Hasilnya: array shape (1, 4) per properti
    # → 1 distance × 4 angles
    fitur = []
    for prop in GLCM_PROPERTIES:
        nilai = graycoprops(glcm, prop)[0]  # ambil distance=1
        # nilai shape: (4,) — 1 nilai per arah
        fitur.extend(nilai.tolist())

    # fitur sekarang berisi 16 nilai:
    # [contrast_0°, contrast_45°, contrast_90°, contrast_135°,
    #  energy_0°, energy_45°, energy_90°, energy_135°,
    #  homogeneity_0°, ... correlation_135°]
    return np.array(fitur, dtype=np.float32)


# ============================================================
# KELAS: Dataset dengan GLCM
# ============================================================
class FERDatasetGLCM(Dataset):
    """
    Custom Dataset yang mengembalikan 2 hal per gambar:
    1. Tensor gambar (untuk MobileNetV2)
    2. Tensor fitur GLCM (untuk cabang GLCM)

    Kenapa perlu custom Dataset?
    Dataset bawaan PyTorch (ImageFolder) hanya mengembalikan
    (gambar, label). Kita butuh (gambar, fitur_glcm, label).
    """

    def __init__(self, root, transform=None, is_train=True):
        # Pakai ImageFolder untuk baca gambar dan label
        self.dataset   = datasets.ImageFolder(root=root)
        self.transform = transform
        self.is_train  = is_train

        # Augmentasi khusus GLCM (ringan, tidak mengubah tekstur terlalu banyak)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]) if is_train else None

        # Cache GLCM supaya tidak dihitung ulang tiap epoch
        # GLCM mahal secara komputasi jika dihitung per batch
        print(f"  Pra-komputasi GLCM untuk {len(self.dataset)} gambar...")
        self.glcm_cache = self._precompute_glcm()
        print(f"  ✅ GLCM cache siap: {len(self.glcm_cache)} sampel")

    def _precompute_glcm(self):
        """Hitung semua fitur GLCM di awal dan simpan di memori."""
        cache = []
        for idx in range(len(self.dataset)):
            path, _ = self.dataset.samples[idx]
            img     = Image.open(path).convert('RGB')
            img     = img.resize((IMG_SIZE, IMG_SIZE))
            fitur   = ekstrak_glcm(img)
            cache.append(fitur)
            if (idx + 1) % 1000 == 0:
                print(f"    {idx+1}/{len(self.dataset)} diproses...", end="\r")
        return cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]

        # Load gambar
        img = Image.open(path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Augmentasi (hanya saat training)
        if self.augment:
            img = self.augment(img)

        # Transform ke tensor
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        # Ambil fitur GLCM dari cache
        glcm_tensor = torch.FloatTensor(self.glcm_cache[idx])

        return img_tensor, glcm_tensor, label

    @property
    def classes(self):
        return self.dataset.classes


# ============================================================
# FUNGSI: Buat DataLoader dengan GLCM
# ============================================================
def buat_dataloader():

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("Memuat dataset training + komputasi GLCM...")
    train_dataset = FERDatasetGLCM(
        root=os.path.join(DATASET_DIR, "train"),
        transform=transform_train,
        is_train=True
    )

    print("\nMemuat dataset test + komputasi GLCM...")
    test_dataset = FERDatasetGLCM(
        root=os.path.join(DATASET_DIR, "test"),
        transform=transform_test,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"\n✅ Dataset siap!")
    print(f"   Train : {len(train_dataset):,} gambar | {len(train_loader)} batch")
    print(f"   Test  : {len(test_dataset):,} gambar | {len(test_loader)} batch")
    print(f"   GLCM  : {GLCM_DIM} fitur per gambar")

    return train_loader, test_loader, train_dataset.classes


# ============================================================
# KELAS: Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss      = nn.functional.cross_entropy(inputs, targets,
                                                    weight=self.alpha,
                                                    reduction='none')
        pt           = torch.exp(-ce_loss)
        focal_loss   = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def hitung_alpha(train_dataset):
    jumlah = [0] * NUM_CLASSES
    for _, _, label in train_dataset:
        jumlah[label] += 1
    total  = sum(jumlah)
    alpha  = [total / (NUM_CLASSES * j) if j > 0 else 0 for j in jumlah]
    rata   = sum(alpha) / len(alpha)
    alpha  = [a / rata for a in alpha]
    print("\nAlpha Focal Loss:")
    for nama, jml, a in zip(train_dataset.classes, jumlah, alpha):
        print(f"  {nama:<12} n={jml:>5,} alpha={a:.3f}")
    return torch.FloatTensor(alpha).to(DEVICE)


# ============================================================
# KELAS: Model Hybrid MobileNetV2 + CNN + GLCM
# ============================================================
class HybridFERModelGLCM(nn.Module):
    """
    Arsitektur Feature Fusion:

    INPUT GAMBAR (128×128×3)
      ↓
    [Cabang A — Deep Learning]
    MobileNetV2 Base (FROZEN)
      ↓ feature map 4×4×1280
    CNN Kustom (Conv256 → Conv128 → AvgPool)
      ↓ 128 dimensi

    INPUT GLCM (16 dimensi)
      ↓
    [Cabang B — GLCM]
    Linear(16 → 32) → ReLU → Linear(32 → 32)
      ↓ 32 dimensi

    CONCATENATE: 128 + 32 = 160 dimensi
      ↓
    [Classifier Gabungan]
    Dense(160→128) → ReLU → Dropout(0.4)
    Dense(128→64)  → ReLU → Dropout(0.3)
    Dense(64→7)    → Softmax
      ↓
    OUTPUT: 7 probabilitas ekspresi

    Kenapa GLCM diproses dulu lewat Linear layer?
    Supaya model bisa "belajar" mana fitur GLCM yang paling
    relevan untuk ekspresi wajah — bukan langsung digabung mentah.
    """

    def __init__(self, num_classes=7, glcm_dim=16):
        super(HybridFERModelGLCM, self).__init__()

        # ---- Cabang A: MobileNetV2 + CNN Kustom ----
        mobilenet = models.mobilenet_v2(pretrained=True)
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
        # Output Cabang A: 128 dimensi

        # ---- Cabang B: GLCM Processing ----
        # Proses fitur GLCM lewat 2 Linear layer
        # supaya model bisa belajar fitur GLCM mana yang penting
        self.glcm_branch = nn.Sequential(
            nn.Linear(glcm_dim, 32),   # 16 → 32
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),          # 32 → 32
            nn.ReLU(inplace=True),
        )
        # Output Cabang B: 32 dimensi

        # ---- Classifier Gabungan ----
        # Menerima concatenation: 128 + 32 = 160 dimensi
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),   # 160 → 128
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),          # 128 → 64
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),  # 64 → 7
            nn.Softmax(dim=1),
        )

    def forward(self, img, glcm):
        """
        img  : tensor gambar (batch, 3, 128, 128)
        glcm : tensor fitur GLCM (batch, 16)
        """

        # ---- Cabang A ----
        x_deep = self.base_model(img)    # (batch, 1280, 4, 4)
        x_deep = self.cnn_kustom(x_deep) # (batch, 128, 1, 1)
        x_deep = x_deep.view(x_deep.size(0), -1)  # (batch, 128)

        # ---- Cabang B ----
        x_glcm = self.glcm_branch(glcm)  # (batch, 32)

        # ---- Concatenate ----
        x_fusi = torch.cat([x_deep, x_glcm], dim=1)  # (batch, 160)

        # ---- Classifier ----
        output = self.classifier(x_fusi)  # (batch, 7)

        return output

    def buka_layer_untuk_finetune(self, n_layer=40):
        semua_layer = list(self.base_model.children())
        total = len(semua_layer)
        for param in self.base_model.parameters():
            param.requires_grad = False
        for layer in semua_layer[total - n_layer:]:
            for param in layer.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameter trainable: {trainable:,}")


# ============================================================
# FUNGSI: 1 epoch training
# ============================================================
def train_satu_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_benar = total_data = 0

    for batch_idx, (gambar, glcm, label) in enumerate(loader):
        gambar = gambar.to(device)
        glcm   = glcm.to(device)
        label  = label.to(device)

        optimizer.zero_grad()
        output = model(gambar, glcm)  # ← kirim 2 input
        loss   = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item() * gambar.size(0)
        pred         = output.argmax(dim=1)
        total_benar += (pred == label).sum().item()
        total_data  += gambar.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}", end="\r")

    return total_loss / total_data, total_benar / total_data


# ============================================================
# FUNGSI: Evaluasi model
# ============================================================
def evaluasi(model, loader, criterion, device):
    model.eval()
    total_loss = total_benar = total_data = 0

    with torch.no_grad():
        for gambar, glcm, label in loader:
            gambar = gambar.to(device)
            glcm   = glcm.to(device)
            label  = label.to(device)
            output = model(gambar, glcm)
            loss   = criterion(output, label)
            total_loss  += loss.item() * gambar.size(0)
            pred         = output.argmax(dim=1)
            total_benar += (pred == label).sum().item()
            total_data  += gambar.size(0)

    return total_loss / total_data, total_benar / total_data


# ============================================================
# FUNGSI: Loop training
# ============================================================
def jalankan_training(model, train_loader, test_loader,
                       optimizer, scheduler, criterion,
                       n_epoch, nama_fase):

    riwayat = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
    akurasi_terbaik = 0.0
    bobot_terbaik   = None

    print(f"\n{'='*50}\n  TRAINING {nama_fase} ({n_epoch} epoch)\n{'='*50}")

    for epoch in range(1, n_epoch + 1):
        t0 = time.time()
        train_loss, train_acc = train_satu_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss,  test_acc  = evaluasi(model, test_loader, criterion, DEVICE)
        scheduler.step(test_loss)

        riwayat["train_loss"].append(train_loss)
        riwayat["train_acc"].append(train_acc)
        riwayat["test_loss"].append(test_loss)
        riwayat["test_acc"].append(test_acc)

        if test_acc > akurasi_terbaik:
            akurasi_terbaik = test_acc
            bobot_terbaik   = copy.deepcopy(model.state_dict())
            torch.save(bobot_terbaik, f"models/best_model_{nama_fase}.pth")

        lr_skrg = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:>3}/{n_epoch} | "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Test Loss:{test_loss:.4f} Acc:{test_acc:.4f} | "
              f"LR:{lr_skrg:.2e} | {time.time()-t0:.1f}s")

    model.load_state_dict(bobot_terbaik)
    print(f"\n✅ {nama_fase} selesai! Akurasi terbaik: {akurasi_terbaik*100:.2f}%")
    return riwayat


# ============================================================
# FUNGSI: Plot grafik
# ============================================================
def plot_history(riwayat, nama_fase):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training {nama_fase} — GLCM + MobileNetV2", fontsize=13)
    ep = range(1, len(riwayat["train_acc"]) + 1)

    ax1.plot(ep, riwayat["train_acc"], color='#378ADD', lw=2, label='Training')
    ax1.plot(ep, riwayat["test_acc"],  color='#D85A30', lw=2, label='Test')
    ax1.set_title('Akurasi per Epoch'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Akurasi')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(ep, riwayat["train_loss"], color='#378ADD', lw=2, label='Training')
    ax2.plot(ep, riwayat["test_loss"],  color='#D85A30', lw=2, label='Test')
    ax2.set_title('Loss per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"models/grafik_{nama_fase}_GLCM.png", dpi=100)
    plt.show()
    print(f"Grafik disimpan: models/grafik_{nama_fase}_GLCM.png")


# ============================================================
# FUNGSI: Evaluasi akhir
# ============================================================
def evaluasi_akhir(model, test_loader, kelas):
    print("\nMengevaluasi model akhir...")
    model.eval()
    semua_pred = []; semua_label = []

    with torch.no_grad():
        for gambar, glcm, label in test_loader:
            gambar = gambar.to(DEVICE)
            glcm   = glcm.to(DEVICE)
            output = model(gambar, glcm)
            pred   = output.argmax(dim=1).cpu().numpy()
            semua_pred.extend(pred)
            semua_label.extend(label.numpy())

    semua_pred  = np.array(semua_pred)
    semua_label = np.array(semua_label)

    print("\n" + "="*55 + "\n  CLASSIFICATION REPORT\n" + "="*55)
    print(classification_report(semua_label, semua_pred,
                                 target_names=[k.capitalize() for k in kelas]))

    cm = confusion_matrix(semua_label, semua_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=[k.capitalize() for k in kelas],
                yticklabels=[k.capitalize() for k in kelas],
                cmap='Blues', linewidths=0.5)
    plt.title('Confusion Matrix — Hybrid GLCM + MobileNetV2', fontsize=13)
    plt.ylabel('Label Asli'); plt.xlabel('Prediksi Model')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig("models/confusion_matrix_GLCM.png", dpi=100)
    plt.show()

    akurasi = np.sum(semua_pred == semua_label) / len(semua_label)
    print(f"\n✅ Akurasi akhir (GLCM + MobileNetV2): {akurasi*100:.2f}%")
    return akurasi


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("="*55)
    print("  STEP 2: TRAINING HYBRID GLCM + MobileNetV2 v4")
    print(f"  IMG_SIZE={IMG_SIZE} | BATCH={BATCH_SIZE}")
    print(f"  GLCM: {len(GLCM_PROPERTIES)} fitur × {len(GLCM_ANGLES)} arah = {GLCM_DIM} dimensi")
    print(f"  Fase1={EPOCH_FASE1} | Fase2={EPOCH_FASE2}")
    print("="*55)

    # Load data
    train_loader, test_loader, kelas = buat_dataloader()

    # Hitung alpha Focal Loss
    alpha     = hitung_alpha(train_loader.dataset)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # Bangun model
    print("\nMembangun model hybrid GLCM + MobileNetV2...")
    model = HybridFERModelGLCM(
        num_classes=NUM_CLASSES,
        glcm_dim=GLCM_DIM
    ).to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameter  : {total:,}")
    print(f"  Dapat dilatih    : {trainable:,}")
    print(f"  Frozen           : {total-trainable:,}  ← MobileNetV2 base")

    # ---- FASE 1 ----
    optimizer_f1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler_f1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_f1, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    riwayat_f1 = jalankan_training(
        model, train_loader, test_loader,
        optimizer_f1, scheduler_f1, criterion,
        EPOCH_FASE1, "Fase1"
    )
    plot_history(riwayat_f1, "Fase1_CNN_frozen")

    # ---- FASE 2 ----
    print("\nMembuka 40 layer terakhir MobileNetV2...")
    model.buka_layer_untuk_finetune(n_layer=40)
    optimizer_f2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )
    scheduler_f2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_f2, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    riwayat_f2 = jalankan_training(
        model, train_loader, test_loader,
        optimizer_f2, scheduler_f2, criterion,
        EPOCH_FASE2, "Fase2"
    )
    plot_history(riwayat_f2, "Fase2_finetune")

    # Simpan model
    torch.save(model.state_dict(), "models/model_glcm_final.pth")
    torch.save(model, "models/model_glcm_full.pth")
    print("\n✅ Model GLCM disimpan: models/model_glcm_full.pth")

    # Evaluasi akhir
    evaluasi_akhir(model, test_loader, kelas)

    print("\n" + "="*55)
    print("  TRAINING SELESAI!")
    print("  Lanjut ke: python 3_realtime_glcm.py")
    print("="*55)
