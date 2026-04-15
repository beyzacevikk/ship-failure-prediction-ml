"""
=============================================================
MODÜL 6: ZİNCİRLEME OLAY ANALİZİ (Failure Propagation)
  Markov Zinciri tabanlı bağımlı olasılık modeli

Ders Notu Gerekçesi:
  Projenin adında "Zincirleme Olay" geçmesinin matematiği burada.
  Temel soru: "Kompresör arızalandıysa, türbin arızası riski ne kadar ARTAR?"

  Bağımlı Koşullu Olasılık:
    P(Türbin Arızası | Kompresör Arızası) → Veri üzerinden tahmin

  Markov Geçiş Matrisi:
    Durum t → Durum t+1 geçiş olasılıklarını öğrenir
    Bu matris sayesinde "4 saat sonra ne olur?" sorusu yanıtlanır.
=============================================================
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. Koşullu Olasılık (Bağımlı Arıza Analizi)
# ─────────────────────────────────────────────
def kosullu_olasilik_hesapla(y_gercek: pd.Series) -> dict:
    """
    Gerçek sınıf etiketlerinden koşullu olasılıkları hesaplar.

    P(Türbin Arızası | Kompresör Arızası) =
        P(Kompresör VE Türbin arızası) / P(Kompresör arızası)

    Bu değer, bağımsız modelin hesaplayamadığı zincirleme riski gösterir.
    """
    n = len(y_gercek)
    p_komp  = (y_gercek == 1).sum() / n
    p_turb  = (y_gercek == 2).sum() / n
    p_normal= (y_gercek == 0).sum() / n

    # Ardışık kayıtlarda birlikte görünme (yakın zaman korelasyonu)
    dizi = y_gercek.values
    komp_sonrasi_turb = 0
    komp_sonrasi_toplam = 0
    for i in range(len(dizi) - 1):
        if dizi[i] == 1:
            komp_sonrasi_toplam += 1
            if dizi[i+1] == 2:
                komp_sonrasi_turb += 1

    p_turb_verilmis_komp = (komp_sonrasi_turb / komp_sonrasi_toplam
                            if komp_sonrasi_toplam > 0 else 0)

    return {
        'p_normal':              p_normal,
        'p_kompresör_arizasi':   p_komp,
        'p_turbin_arizasi':      p_turb,
        'p_turb_verilmis_komp':  p_turb_verilmis_komp,
        'komp_sonrasi_toplam':   komp_sonrasi_toplam,
        'komp_sonrasi_turb':     komp_sonrasi_turb,
    }


# ─────────────────────────────────────────────
# 2. Markov Geçiş Matrisi
# ─────────────────────────────────────────────
def markov_gecis_matrisi_olustur(y_gercek: pd.Series) -> np.ndarray:
    """
    Ardışık gözlemlerden Markov geçiş matrisini öğrenir.

    Matris[i][j] = P(Durum_t+1 = j | Durum_t = i)

    Örnek: Matris[1][2] = 0.23 →
           "Kompresör arızasından sonra Türbin arızası olasılığı %23"
    """
    n_sinif = 3
    sayim = np.zeros((n_sinif, n_sinif))
    dizi = y_gercek.values
    for t in range(len(dizi) - 1):
        mevcut = int(dizi[t])
        sonraki = int(dizi[t + 1])
        if 0 <= mevcut < n_sinif and 0 <= sonraki < n_sinif:
            sayim[mevcut][sonraki] += 1

    # Normalize et: satır toplamı = 1
    toplam = sayim.sum(axis=1, keepdims=True)
    toplam[toplam == 0] = 1
    return sayim / toplam


def markov_matrisi_yazdir(matris: np.ndarray):
    """Geçiş matrisini okunaklı tablo olarak yazdırır."""
    siniflar = ['Normal', 'Komp.Arıza', 'Türb.Arıza']
    print("  Markov Geçiş Matrisi  P(Sonraki Durum | Mevcut Durum):")
    print(f"  {'Mevcut → Sonraki':20} {'Normal':>12} {'Komp.Arıza':>12} {'Türb.Arıza':>12}")
    print("  " + "─" * 58)
    for i, satir_isim in enumerate(siniflar):
        degerler = [f"%{matris[i][j]*100:6.1f}" for j in range(3)]
        vurgu = ""
        if i == 1:  # Kompresör sonrası türbin geçiş olasılığı
            vurgu = f"  ← %{matris[i][2]*100:.1f} zincir riski" if matris[i][2] > 0.05 else ""
        print(f"  {satir_isim:<20} {degerler[0]:>12} {degerler[1]:>12} {degerler[2]:>12}{vurgu}")


def n_adim_ileri_tahmini(matris: np.ndarray, baslangic_sinifi: int, n_adim: int = 4) -> np.ndarray:
    """
    Markov matrisini n kez çarparak n adım sonraki olasılık dağılımını hesaplar.

    Örnek: Kompresör arızasından 4 ölçüm sonra Türbin Arızası olasılığı nedir?
           matris^4 ile yanıtlanır.
    """
    n_sinif = matris.shape[0]
    mevcut_durum = np.zeros(n_sinif)
    mevcut_durum[baslangic_sinifi] = 1.0  # Başlangıç: bu sınıfta kesin

    for _ in range(n_adim):
        mevcut_durum = mevcut_durum @ matris

    return mevcut_durum


# ─────────────────────────────────────────────
# 3. Dinamik Risk Raporu (Senaryo)
# ─────────────────────────────────────────────
def dinamik_risk_raporu(model, X_test_olcekli, y_test: pd.Series,
                         ozellik_isimleri: list, gecis_matrisi: np.ndarray):
    """
    En yüksek riskli test örnekleri için tam zincirleme senaryo raporu üretir.

    Bu rapor, gemi başmühendisine sunulan Dinamik Risk Raporu'dur:
      Sensör değerleri → Model tahmini → Zincirleme risk → Aksiyon planı
    """
    sinif_isimleri = {0: 'Normal', 1: 'Kompresör Arızası', 2: 'Türbin Arızası'}
    tahminler     = model.predict(X_test_olcekli)
    olasiliklar   = model.predict_proba(X_test_olcekli)
    risk_skorlari = (1 - olasiliklar[:, 0]) * 100

    # En yüksek riskli 5 örneği seç
    en_yuksek_idx = np.argsort(risk_skorlari)[::-1][:5]

    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║   DİNAMİK RİSK RAPORU — GEMİ MAKİNE DAİRESİ ERKEN UYARI       ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    for sira, idx in enumerate(en_yuksek_idx, 1):
        tahmin      = tahminler[idx]
        gercek      = y_test.iloc[idx]
        risk        = risk_skorlari[idx]
        prob_vektor = olasiliklar[idx]
        dogru_mu    = "✔ DOĞRU" if tahmin == gercek else "✗ YANLIŞ"

        print()
        print(f"  ── UYARI #{sira} {'─'*48}")
        print(f"  Tespit Edilen Durum  : {sinif_isimleri[tahmin]}  [{dogru_mu}]")
        print(f"  Gerçek Durum         : {sinif_isimleri[gercek]}")
        print(f"  Genel Risk Skoru     : %{risk:.1f}")
        print(f"  Sınıf Olasılıkları   : Normal=%{prob_vektor[0]*100:.1f}  "
              f"Komp.Arıza=%{prob_vektor[1]*100:.1f}  "
              f"Türb.Arıza=%{prob_vektor[2]*100:.1f}")

        # Zincirleme: Mevcut durumdan 4 adım sonrası
        zincir_4 = n_adim_ileri_tahmini(gecis_matrisi, tahmin, n_adim=4)
        print(f"  Zincirleme Tahmin    : 4 ölçüm sonrası olasılıklar:")
        print(f"    Normal=%{zincir_4[0]*100:.1f}  Komp.Arıza=%{zincir_4[1]*100:.1f}  "
              f"Türb.Arıza=%{zincir_4[2]*100:.1f}")

        # Zincir uyarısı
        if tahmin == 1 and gecis_matrisi[1][2] > 0.05:
            turb_riski = gecis_matrisi[1][2] * 100
            print(f"  ⚠ ZİNCİR UYARISI: Kompresör arızası → %{turb_riski:.1f} olasılıkla")
            print(f"    Türbin Çıkış Sıcaklığı emniyet limitini (650°C) aşabilir.")

        # Risk seviyesi + Aksiyon Planı
        if risk >= 75:
            seviye = "🚨 KRİTİK"
            aksiyon = [
                "1. Makine dairesine bildir — anlık müdahale gerekli",
                "2. Yedek kompresöre geçişi hazırla",
                "3. Yakıt akışını %10 kısıtla",
                "4. Köprüden hız düşürme talebi ilet",
            ]
        elif risk >= 50:
            seviye = "⚠ YÜKSEK"
            aksiyon = [
                "1. Kompresör filtresini kontrol et",
                "2. Sensör değerlerini 15 dakikada bir izle",
                "3. Bakım mühendisini bildir",
            ]
        else:
            seviye = "✅ NORMAL"
            aksiyon = ["1. Rutin izleme devam ediyor."]

        print(f"  Risk Seviyesi        : {seviye}")
        print(f"  Aksiyon Planı:")
        for a in aksiyon:
            print(f"    {a}")

    return risk_skorlari, tahminler


# ─────────────────────────────────────────────
# Ana Zincir Analizi Fonksiyonu
# ─────────────────────────────────────────────
def zincir_analizi_yap(model, X_test_olcekli, y_test: pd.Series,
                        y_egitim: pd.Series, ozellik_isimleri: list) -> dict:
    """
    Tam zincirleme analizi pipeline'ı.
    """
    print("=" * 60)
    print("  [6/6] ZİNCİRLEME OLAY ANALİZİ (Failure Propagation)")
    print("=" * 60)
    print()

    # Koşullu olasılıklar
    print("  Koşullu Olasılık Analizi (Bağımlı Arıza Modeli):")
    kop = kosullu_olasilik_hesapla(y_egitim)
    print(f"    P(Normal)                          : %{kop['p_normal']*100:.1f}")
    print(f"    P(Kompresör Arızası)               : %{kop['p_kompresör_arizasi']*100:.1f}")
    print(f"    P(Türbin Arızası)                  : %{kop['p_turbin_arizasi']*100:.1f}")
    print(f"    P(Türbin Arızası | Komp. Arızası)  : %{kop['p_turb_verilmis_komp']*100:.1f}")
    print()
    if kop['p_turb_verilmis_komp'] > 0:
        oran = kop['p_turb_verilmis_komp'] / kop['p_turbin_arizasi']
        print(f"    → Kompresör arızası varken Türbin arızası riski {oran:.1f}x ARTIYOR")
    print()

    # Markov matrisi
    gecis_matrisi = markov_gecis_matrisi_olustur(y_egitim)
    markov_matrisi_yazdir(gecis_matrisi)
    print()

    # Dinamik risk raporu
    risk_skorlari, tahminler = dinamik_risk_raporu(
        model, X_test_olcekli, y_test, ozellik_isimleri, gecis_matrisi
    )

    return {
        'kosullu_olasiliklar': kop,
        'gecis_matrisi':       gecis_matrisi,
        'risk_skorlari':       risk_skorlari,
        'tahminler':           tahminler,
    }
