import os

EKSPRESI = ['angry','disgust','fear','happy','neutral','sad','surprise']
total = 0

print(f"{'Ekspresi':<12} {'Jumlah':>8}")
print("-" * 22)
for e in EKSPRESI:
    n = len(os.listdir(f"data_latih/dataset/{e}"))
    total += n
    print(f"{e:<12} {n:>8,}")
print("-" * 22)
print(f"{'TOTAL':<12} {total:>8,}")