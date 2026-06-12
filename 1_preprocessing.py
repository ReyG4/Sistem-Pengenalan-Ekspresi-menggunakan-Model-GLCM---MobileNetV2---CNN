"""
FILE: 1_preprocessing.py  — v3
PERUBAHAN: IMG_SIZE 96 → 128 (lebih detail, tetap cepat)
"""

import os, shutil, glob, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

DATASET_DIR = "data_latih/dataset"
OUTPUT_DIR  = "dataset_split"
IMG_SIZE    = 128          # ← naik dari 96
BATCH_SIZE  = 64
RANDOM_SEED = 42
SPLIT_TEST  = 0.20
EKSPRESI    = ['angry','disgust','fear','happy','neutral','sad','surprise']


def cek_dataset():
    print("="*50 + "\n  CEK DATASET\n" + "="*50)
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Folder '{DATASET_DIR}' tidak ditemukan!")

    print(f"{'Ekspresi':<12} {'Jumlah':>8}")
    print("-"*22)
    total, jumlah_per_kelas = 0, {}

    for e in EKSPRESI:
        folder = os.path.join(DATASET_DIR, e)
        if not os.path.exists(folder):
            print(f"{e:<12} {'TIDAK ADA':>8} ⚠️")
            jumlah_per_kelas[e] = 0
            continue
        gambar = (glob.glob(os.path.join(folder,"*.jpg")) +
                  glob.glob(os.path.join(folder,"*.jpeg")) +
                  glob.glob(os.path.join(folder,"*.png")))
        n = len(gambar)
        jumlah_per_kelas[e] = n
        total += n
        print(f"{e:<12} {n:>8,}")

    print("-"*22)
    print(f"{'TOTAL':<12} {total:>8,}")

    nilai = list(jumlah_per_kelas.values())
    if max(nilai) > 0 and min(nilai) > 0:
        rasio = max(nilai)/min(nilai)
        status = "⚠️ Cukup timpang" if rasio > 5 else ("⚠️ Sedikit timpang" if rasio > 3 else "✅ Seimbang")
        print(f"\nRasio max:min = {rasio:.1f}x {status}")

    print(f"\nKonfigurasi:")
    print(f"  IMG_SIZE   : {IMG_SIZE}x{IMG_SIZE} px")
    print(f"  BATCH_SIZE : {BATCH_SIZE}")
    print(f"  Split      : {int((1-SPLIT_TEST)*100)}% train | {int(SPLIT_TEST*100)}% test")
    print(f"  Seed       : {RANDOM_SEED}")
    return jumlah_per_kelas


def split_dataset():
    print("\n" + "="*50 + "\n  SPLIT DATASET\n" + "="*50)
    if os.path.exists(OUTPUT_DIR):
        print(f"Menghapus folder lama: {OUTPUT_DIR}/")
        shutil.rmtree(OUTPUT_DIR)

    total_train = total_test = 0
    for e in EKSPRESI:
        src = os.path.join(DATASET_DIR, e)
        if not os.path.exists(src):
            print(f"⚠️  Skip {e} — folder tidak ada")
            continue

        semua = (glob.glob(os.path.join(src,"*.jpg")) +
                 glob.glob(os.path.join(src,"*.jpeg")) +
                 glob.glob(os.path.join(src,"*.png")))
        if not semua:
            print(f"⚠️  Skip {e} — tidak ada gambar")
            continue

        train_f, test_f = train_test_split(semua, test_size=SPLIT_TEST,
                                            random_state=RANDOM_SEED, shuffle=True)

        for subdir, files in [(os.path.join(OUTPUT_DIR,"train",e), train_f),
                               (os.path.join(OUTPUT_DIR,"test",e),  test_f)]:
            os.makedirs(subdir, exist_ok=True)
            for f in files:
                shutil.copy(f, subdir)

        total_train += len(train_f)
        total_test  += len(test_f)
        print(f"{e:<12} total={len(semua):>5,} | train={len(train_f):>5,} | test={len(test_f):>5,}")

    print("-"*50)
    print(f"{'TOTAL':<12} train={total_train:>5,} | test={total_test:>5,}")
    print(f"\n✅ Split selesai! Hasil di folder: {OUTPUT_DIR}/")


def visualisasi_sampel():
    print("\nMenampilkan sampel gambar...")
    fig, axes = plt.subplots(3, 7, figsize=(18, 8))
    fig.suptitle(f"Sampel Dataset — IMG_SIZE:{IMG_SIZE}px | Seed:{RANDOM_SEED} | Split:{int((1-SPLIT_TEST)*100)}/{int(SPLIT_TEST*100)}", fontsize=13)
    train_dir = os.path.join(OUTPUT_DIR, "train")

    for col, e in enumerate(EKSPRESI):
        folder = os.path.join(train_dir, e)
        imgs = (glob.glob(os.path.join(folder,"*.jpg")) +
                glob.glob(os.path.join(folder,"*.jpeg")) +
                glob.glob(os.path.join(folder,"*.png")))
        np.random.seed(RANDOM_SEED)
        sampel = np.random.choice(imgs, size=min(3,len(imgs)), replace=False)
        for row, path in enumerate(sampel):
            ax = axes[row][col]
            ax.imshow(Image.open(path).convert('RGB'))
            if row == 0: ax.set_title(e.capitalize(), fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/sampel_dataset.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("✅ Sampel disimpan: models/sampel_dataset.png")


def simpan_konfigurasi(jumlah_per_kelas):
    os.makedirs("models", exist_ok=True)
    config = {
        "dataset_dir": OUTPUT_DIR, "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE, "random_seed": RANDOM_SEED,
        "split_test": SPLIT_TEST, "ekspresi": EKSPRESI,
        "jumlah_per_kelas": jumlah_per_kelas,
    }
    with open("models/config.json","w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Konfigurasi disimpan: models/config.json")
    print(f"   IMG_SIZE={IMG_SIZE} | BATCH={BATCH_SIZE} | SEED={RANDOM_SEED} | SPLIT={int((1-SPLIT_TEST)*100)}/{int(SPLIT_TEST*100)}")


if __name__ == "__main__":
    print("="*50 + f"\n  STEP 1: PREPROCESSING v3 — IMG_SIZE={IMG_SIZE}\n" + "="*50)
    jumlah_per_kelas = cek_dataset()
    split_dataset()
    visualisasi_sampel()
    simpan_konfigurasi(jumlah_per_kelas)
    print("\n" + "="*50 + "\n  PREPROCESSING SELESAI!\n  Lanjut ke: python 2_train.py\n" + "="*50)