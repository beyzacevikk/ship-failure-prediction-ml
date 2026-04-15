"""
=============================================================
MODÜL: KEŞİFSEL VERİ ANALİZİ (EDA)
Pipeline Adım 3 — Veri Analitiği
  - Dağılım analizi (histogram)
  - Korelasyon matrisi (hedef değişkenle ilişki)
  - Aykırı değer tespiti (IQR + kutu grafiği)
  - Hedef değişken dağılımı (aşınma katsayıları)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'

HEDEF_SUTUNLAR = ['kompresor_asinma_katsayisi', 'turbin_asinma_katsayisi']
SINIF_RENKLERI = {'Normal': '#2ecc71', 'Komp.Arızası': '#e74c3c', 'Türb.Arızası': '#e67e22'}


def eda_rapor(veri: pd.DataFrame, kayit_klasoru: str = '.'):
    """Tam EDA pipeline'ı — konsol + görsel çıktı."""
    print("=" * 60)
    print("  [EDA] KEŞİFSEL VERİ ANALİZİ")
    print("=" * 60)

    # ── 1. Temel İstatistikler ─────────────────
    print("\n  1) Temel İstatistikler:")
    desc = veri.describe().T[['mean', 'std', 'min', '50%', 'max']]
    for col, row in desc.iterrows():
        print(f"    {col:<40}: ort={row['mean']:>10.4f}  std={row['std']:>10.4f}"
              f"  min={row['min']:>10.4f}  max={row['max']:>10.4f}")

    # ── 2. Eksik Veri ─────────────────────────
    print("\n  2) Eksik Veri Analizi:")
    eksik = veri.isnull().sum()
    if eksik.sum() == 0:
        print("    ✔ Eksik değer yok — veri seti temiz.")
    else:
        for col, n in eksik[eksik > 0].items():
            print(f"    ⚠ {col}: {n} eksik değer (%{n/len(veri)*100:.2f})")

    # ── 3. Aykırı Değer Raporu ─────────────────
    print("\n  3) IQR Aykırı Değer Tespiti (≥1% oranında):")
    sensor_sutunlar = [c for c in veri.columns if c not in HEDEF_SUTUNLAR]
    for col in sensor_sutunlar:
        Q1, Q3 = veri[col].quantile(0.25), veri[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        aykiri = ((veri[col] < Q1 - 1.5 * IQR) | (veri[col] > Q3 + 1.5 * IQR)).sum()
        oran = aykiri / len(veri) * 100
        if oran >= 1:
            sembol = "⚠" if oran < 5 else "🚨"
            print(f"    {sembol} {col:<38}: {aykiri} aykırı değer (%{oran:.1f})")

    # ── 4. Korelasyon Analizi ──────────────────
    print("\n  4) Hedef Değişkenlerle Yüksek Korelasyon (|r| > 0.5):")
    for hedef in HEDEF_SUTUNLAR:
        korelasyon = veri[sensor_sutunlar].corrwith(veri[hedef]).abs().sort_values(ascending=False)
        yuksek = korelasyon[korelasyon > 0.5]
        print(f"    → {hedef}:")
        for col, r in yuksek.items():
            print(f"       {col:<38}: r={r:.3f}")
        if len(yuksek) == 0:
            print("       (0.5 üstü korelasyon bulunamadı)")

    # ── Görseller ─────────────────────────────
    _dagilim_grafigi(veri, sensor_sutunlar, kayit_klasoru)
    _korelasyon_haritasi(veri, kayit_klasoru)
    _hedef_dagilim_grafigi(veri, kayit_klasoru)

    print("\n  ✔ EDA tamamlandı.")


def _dagilim_grafigi(veri: pd.DataFrame, sutunlar: list, kayit_klasoru: str):
    """Sensör histogramları + KDE."""
    n = min(len(sutunlar), 16)
    secilen = sutunlar[:n]
    cols = 4
    rows = (n + cols - 1) // cols

    fig, eksenler = plt.subplots(rows, cols, figsize=(16, rows * 3))
    eksenler = eksenler.flatten()

    for i, col in enumerate(secilen):
        eksenler[i].hist(veri[col], bins=40, color='#3498db', alpha=0.7, edgecolor='white')
        eksenler[i].set_title(col, fontsize=8, fontweight='bold')
        eksenler[i].set_xlabel('Değer', fontsize=7)
        eksenler[i].set_ylabel('Frekans', fontsize=7)
        eksenler[i].tick_params(labelsize=6)
        eksenler[i].grid(alpha=0.3)

    for j in range(i + 1, len(eksenler)):
        eksenler[j].set_visible(False)

    plt.suptitle('Sensör Dağılımları — Histogram', fontsize=13, fontweight='bold')
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/eda_dagilimlar.png'
    plt.savefig(dosya, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def _korelasyon_haritasi(veri: pd.DataFrame, kayit_klasoru: str):
    """Korelasyon ısı haritası — hedef değişkenlerle birlikte."""
    sayisal = veri.select_dtypes(include=[np.number])
    corr = sayisal.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                annot=False, linewidths=0.3, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Özellik Korelasyon Matrisi\n'
                 '(Kırmızı = Pozitif, Mavi = Negatif korelasyon)',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/eda_korelasyon.png'
    plt.savefig(dosya, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def _hedef_dagilim_grafigi(veri: pd.DataFrame, kayit_klasoru: str):
    """Hedef aşınma katsayılarının dağılımı."""
    fig, eksenler = plt.subplots(1, 2, figsize=(12, 5))

    for ax, hedef, renk in zip(eksenler, HEDEF_SUTUNLAR, ['#e74c3c', '#e67e22']):
        ax.hist(veri[hedef], bins=50, color=renk, alpha=0.75, edgecolor='white')
        ax.axvline(veri[hedef].mean(), color='black', linestyle='--',
                   linewidth=1.5, label=f'Ort: {veri[hedef].mean():.4f}')
        ax.axvline(veri[hedef].quantile(0.33), color='blue', linestyle=':',
                   linewidth=1.5, label=f'33. Persentil (arıza eşiği)')
        ax.set_title(f'{hedef}\nDağılım Analizi', fontsize=11, fontweight='bold')
        ax.set_xlabel('Aşınma Katsayısı', fontsize=10)
        ax.set_ylabel('Kayıt Sayısı', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Hedef Değişkenler — Aşınma Katsayısı Dağılımları\n'
                 '(Eşik altı → Arıza sınıfı)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/eda_hedef_dagilim.png'
    plt.savefig(dosya, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")
