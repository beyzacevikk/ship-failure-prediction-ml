"""
=============================================================
MODÜL 2: VERİ ÖN İŞLEME (GELİŞMİŞ)
Ders Notu İlkeleri:
  - Varyansı sıfır olan özellikler bilgi taşımaz (Entropi = 0)
  - IQR yöntemiyle aykırı değer TESPİT + BASKILA (Winsorization)
  - Özellik Mühendisliği: Hareketli ortalama + zaman serisi farkları
  - Mesafe bazlı algoritmalar için StandardScaler normalizasyonu
  - Stratified split + Data Leakage önlemi
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SABIT_SUTUNLAR = ['komp_giris_sicaklik', 'komp_giris_basinc']


def sabit_sutunlari_kaldir(veri: pd.DataFrame) -> pd.DataFrame:
    """
    Varyansı sıfır olan sütunları kaldırır.
    Ders Notu: Bilgi Teorisi → Entropi = 0 ise özellik bilgi taşımaz.
    """
    print("  [2a] Sabit sütun temizliği:")
    gercek_sabitler = [s for s in veri.columns if veri[s].nunique() == 1]
    tum_kaldir = list(set(SABIT_SUTUNLAR) | set(gercek_sabitler))
    for s in tum_kaldir:
        print(f"       '{s}' → {veri[s].nunique()} benzersiz değer → ÇIKARILIYOR")
    return veri.drop(columns=tum_kaldir)


def aykiri_degerleri_baski(veri: pd.DataFrame, sutunlar: list) -> pd.DataFrame:
    """
    IQR yöntemiyle aykırı değerleri tespit eder ve sınır değerlere BASKILAR (clip).

    Ders Notu Gerekçesi:
        Sadece raporlamak yetmez. Aykırı değerleri SİLMİYORUZ (veri kaybı olur),
        BASKILIYOR (Winsorization / clipping) yapıyoruz:
          alt sınır altındaki → alt sınıra çekil
          üst sınır üstündeki → üst sınıra çekil
    """
    print("  [2b] IQR Aykırı Değer Baskılama (Winsorization):")
    veri = veri.copy()
    toplam = 0
    for sutun in sutunlar:
        Q1, Q3 = veri[sutun].quantile(0.25), veri[sutun].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        alt, ust = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        aykiri = ((veri[sutun] < alt) | (veri[sutun] > ust)).sum()
        if aykiri > 0:
            veri[sutun] = veri[sutun].clip(lower=alt, upper=ust)
            toplam += aykiri
            print(f"       '{sutun}': {aykiri} değer baskılandı → [{alt:.3f}, {ust:.3f}]")
    if toplam == 0:
        print("       Tüm değerler IQR sınırları içinde — baskılama gerekmedi.")
    else:
        print(f"       Toplam baskılanan: {toplam} değer")
    return veri


def ozellik_muhendisligi(veri: pd.DataFrame) -> pd.DataFrame:
    """
    Ham sensör değerlerine değişim bilgisi ekler.

    Ders Notu Gerekçesi:
        Arıza sadece değerin büyüklüğüyle değil, DEĞİŞİM HIZI ile ilgilidir.
          _hort5 : Son 5 ölçümün hareketli ortalaması → trend yumuşatma
          _fark1 : Bir önceki ölçüme göre fark → anlık değişim hızı
        Örnek: Basınç 1.05'ten 1.03'e düşmesi, modelin "düşüş başladı"
               demesini sağlar — ham değer yeterince bilgi vermez.
    """
    print("  [2c] Özellik Mühendisliği (Hareketli Ort. + Fark):")
    kritik = ['komp_cikis_basinc', 'komp_cikis_sicaklik',
              'hp_turbin_cikis_sicakligi', 'hp_turbin_cikis_basinci',
              'yakit_akisi', 'gt_devir']
    mevcut = [s for s in kritik if s in veri.columns]
    veri = veri.copy()
    yeni = []
    for sutun in mevcut:
        ad_ort  = f'{sutun}_hort5'
        ad_fark = f'{sutun}_fark1'
        veri[ad_ort]  = veri[sutun].rolling(window=5, min_periods=1).mean()
        veri[ad_fark] = veri[sutun].diff().fillna(0)
        yeni += [ad_ort, ad_fark]
    print(f"       {len(yeni)} yeni özellik türetildi (6 sensör × 2 dönüşüm).")
    print(f"       Toplam özellik sayısı: {veri.shape[1]}")
    return veri


def etiketle(veri: pd.DataFrame) -> pd.DataFrame:
    """
    Aşınma katsayılarından 3 sınıflı etiket üretir (Multiclass Classification).
    Sınıf 0 = Normal | 1 = Kompresör Arızası | 2 = Türbin Arızası
    """
    print("  [2d] Çok Sınıflı Etiketleme:")
    komp_esik = veri['kompresor_asinma_katsayisi'].quantile(0.33)
    turb_esik = veri['turbin_asinma_katsayisi'].quantile(0.33)
    print(f"       Kompresör eşik : {komp_esik:.4f}  |  Türbin eşik : {turb_esik:.4f}")

    def _sinif(satir):
        if satir['kompresor_asinma_katsayisi'] < komp_esik:
            return 1
        elif satir['turbin_asinma_katsayisi'] < turb_esik:
            return 2
        return 0

    veri = veri.copy()
    veri['ariza_sinifi'] = veri.apply(_sinif, axis=1)
    isimler = {0: 'Normal', 1: 'Kompresör Arızası', 2: 'Türbin Arızası'}
    for s, n in veri['ariza_sinifi'].value_counts().sort_index().items():
        print(f"         Sınıf {s} ({isimler[s]}): {n:,} (%{n/len(veri)*100:.1f})")
    return veri


def on_isle(veri: pd.DataFrame) -> tuple:
    """Ana ön işleme pipeline'ı."""
    print("=" * 60)
    print("  [2/5] VERİ ÖN İŞLEME")
    print("=" * 60)

    veri = sabit_sutunlari_kaldir(veri);  print()
    sensor_sutunlar = [s for s in veri.columns
                       if s not in ['kompresor_asinma_katsayisi', 'turbin_asinma_katsayisi']]
    veri = aykiri_degerleri_baski(veri, sensor_sutunlar); print()
    veri = ozellik_muhendisligi(veri);    print()
    veri = etiketle(veri);                print()

    hedef_ve_katsayilar = ['ariza_sinifi', 'kompresor_asinma_katsayisi', 'turbin_asinma_katsayisi']
    X = veri.drop(columns=hedef_ve_katsayilar)
    y = veri['ariza_sinifi']
    ozellik_isimleri = X.columns.tolist()

    # Stratified split: sınıf oranları korunur
    X_egitim, X_test, y_egitim, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # Data Leakage önlemi: fit() SADECE eğitim setine
    olceklendirici = StandardScaler()
    X_egitim_olcekli = olceklendirici.fit_transform(X_egitim)
    X_test_olcekli   = olceklendirici.transform(X_test)   # sadece transform!

    print(f"  ✔ Eğitim: {X_egitim.shape[0]:,} | Test: {X_test.shape[0]:,} | Özellik: {X_egitim_olcekli.shape[1]}")
    print(f"  ✔ Normalizasyon: StandardScaler (fit → yalnızca eğitim)")
    return (X_egitim_olcekli, X_test_olcekli, y_egitim, y_test, olceklendirici, ozellik_isimleri)
