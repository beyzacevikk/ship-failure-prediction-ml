"""
=============================================================
MODÜL 5: GÖRSELLEŞTİRME (GELİŞMİŞ)
  - Confusion Matrix ısı haritası (normalize + sayılar)
  - Model karşılaştırma çubuk grafiği
  - Özellik önem grafiği (ham + türetilmiş özellikler renk kodlu)
  - Risk skoru dağılımı paneli
  - Markov geçiş matrisi görselleştirmesi
  - Dashboard: Sensör Grubu → Risk → Tahmin → Aksiyon
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'

SINIF_ISIMLER = ['Normal', 'Komp.Arızası', 'Türb.Arızası']
RENKLER_SINIF = ['#2ecc71', '#e74c3c', '#e67e22']


def karisiklik_matrisi_ciz(metrikler_listesi: list, kayit_klasoru: str = '.'):
    """Normalize edilmiş confusion matrix ısı haritası."""
    n = len(metrikler_listesi)
    fig, eksenler = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        eksenler = [eksenler]

    for ax, m in zip(eksenler, metrikler_listesi):
        cm = m['karisiklik_matrisi']
        toplam_satir = cm.sum(axis=1, keepdims=True)
        toplam_satir[toplam_satir == 0] = 1
        cm_norm = cm / toplam_satir

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    xticklabels=SINIF_ISIMLER, yticklabels=SINIF_ISIMLER,
                    linewidths=0.5, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                renk = '#ff4444' if (i != j and cm[i, j] > 0) else 'gray'
                ax.text(j + 0.5, i + 0.78, f'n={cm[i,j]}',
                        ha='center', va='center', fontsize=7, color=renk)

        ax.set_title(f'{m["model_adi"]}\nRecall=%{m["duyarlilik"]*100:.1f} | Acc=%{m["dogruluk"]*100:.1f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('TAHMİN', fontsize=9)
        ax.set_ylabel('GERÇEK', fontsize=9)

    plt.suptitle('Karmaşıklık Matrisi — Normalize Oranlar\n(Köşegen = Doğru tahmin)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/karisiklik_matrisi.png'
    plt.savefig(dosya, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def model_karsilastirma_ciz(metrikler_listesi: list, kayit_klasoru: str = '.'):
    """Model metriklerini yan yana çubuk grafik."""
    model_adlari = [m['model_adi'] for m in metrikler_listesi]
    metrik_anahtarlar = ['dogruluk', 'duyarlilik', 'kesinlik', 'f1_skor']
    metrik_etiketler = ['Doğruluk', 'Duyarlılık\n(Recall)', 'Kesinlik\n(Precision)', 'F1 Skor']
    renkler = ['#3498db', '#e74c3c', '#2ecc71']

    x = np.arange(len(metrik_anahtarlar))
    genislik = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model_adi, renk) in enumerate(zip(model_adlari, renkler)):
        degerler = [metrikler_listesi[i][k] * 100 for k in metrik_anahtarlar]
        cubuklar = ax.bar(x + i * genislik, degerler, genislik,
                          label=model_adi, color=renk, alpha=0.85, edgecolor='white')
        for c in cubuklar:
            ax.annotate(f'%{c.get_height():.1f}',
                        xy=(c.get_x() + c.get_width() / 2, c.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Recall vurgu bandı
    ax.axvspan(0.5, 1.4, alpha=0.07, color='orange')
    ax.text(0.93, 73, '★ Gemi\nEmniyeti', fontsize=8, color='darkorange',
            style='italic', ha='center')

    ax.set_xlabel('Performans Metrikleri', fontsize=11)
    ax.set_ylabel('Skor (%)', fontsize=11)
    ax.set_title('Algoritma Karşılaştırması — GridSearchCV Optimize Edilmiş Modeller',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x + genislik)
    ax.set_xticklabels(metrik_etiketler, fontsize=10)
    ax.set_ylim(60, 105)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/model_karsilastirma.png'
    plt.savefig(dosya, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def ozellik_onemliligi_ciz(model, ozellik_isimleri: list, kayit_klasoru: str = '.'):
    """Özellik önem grafiği — ham vs türetilmiş özellik renk kodu."""
    if not hasattr(model, 'feature_importances_'):
        return

    onemler = model.feature_importances_
    indeksler = np.argsort(onemler)[::-1][:15]
    top_isim = [ozellik_isimleri[i] for i in indeksler]
    top_onem = onemler[indeksler]

    def renk_sec(isim):
        if '_hort5' in isim:
            return '#9b59b6'    # Mor: hareketli ortalama
        if '_fark1' in isim:
            return '#1abc9c'    # Turkuaz: fark (değişim hızı)
        if 'komp' in isim or 'basinc' in isim:
            return '#e74c3c'    # Kırmızı: kompresör
        if 'turbin' in isim or 'sicaklik' in isim:
            return '#e67e22'    # Turuncu: türbin
        return '#3498db'        # Mavi: gemi operasyonu

    renkler_bar = [renk_sec(n) for n in top_isim]

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(top_isim))
    ax.barh(y_pos, top_onem * 100, color=renkler_bar, alpha=0.85, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_isim, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Önem Skoru (%)', fontsize=11)
    ax.set_title('Özellik Önem Sıralaması (Rastgele Orman)\n'
                 'Ham Sensörler + Özellik Mühendisliği Karşılaştırması',
                 fontsize=12, fontweight='bold')

    legend_eleman = [
        mpatches.Patch(color='#e74c3c', label='Ham — Kompresör'),
        mpatches.Patch(color='#e67e22', label='Ham — Türbin/Sıcaklık'),
        mpatches.Patch(color='#3498db', label='Ham — Gemi Operasyonu'),
        mpatches.Patch(color='#9b59b6', label='Türetilmiş — Hareketli Ort.'),
        mpatches.Patch(color='#1abc9c', label='Türetilmiş — Değişim Hızı'),
    ]
    ax.legend(handles=legend_eleman, loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/ozellik_onemliligi.png'
    plt.savefig(dosya, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def markov_matrisi_ciz(gecis_matrisi: np.ndarray, kayit_klasoru: str = '.'):
    """Markov geçiş matrisini heatmap ve ok diyagramıyla gösterir."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: Heatmap
    sns.heatmap(gecis_matrisi * 100, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=SINIF_ISIMLER, yticklabels=SINIF_ISIMLER,
                ax=ax1, linewidths=0.5, vmin=0, vmax=100)
    ax1.set_title('Markov Geçiş Matrisi\nP(Sonraki | Mevcut) — %', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Sonraki Durum', fontsize=10)
    ax1.set_ylabel('Mevcut Durum', fontsize=10)

    # Sağ: N-adım ileri tahmin (1 → 4 adım)
    from zincir_analizi import n_adim_ileri_tahmini
    adimlar = list(range(1, 9))
    siniflar_zaman = {isim: [] for isim in SINIF_ISIMLER}

    for adim in adimlar:
        dist = n_adim_ileri_tahmini(gecis_matrisi, baslangic_sinifi=1, n_adim=adim)
        for j, isim in enumerate(SINIF_ISIMLER):
            siniflar_zaman[isim].append(dist[j] * 100)

    renkler_zaman = ['#2ecc71', '#e74c3c', '#e67e22']
    for isim, renk in zip(SINIF_ISIMLER, renkler_zaman):
        ax2.plot(adimlar, siniflar_zaman[isim], 'o-', color=renk, label=isim, linewidth=2)

    ax2.set_xlabel('İlerleyen Ölçüm Adımı', fontsize=10)
    ax2.set_ylabel('Olasılık (%)', fontsize=10)
    ax2.set_title('Zincirleme İlerleme\nBaşlangıç: Kompresör Arızası → Sonraki Durumlar',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(adimlar)
    ax2.set_xticklabels([f't+{a}' for a in adimlar])

    plt.suptitle('ZİNCİRLEME OLAY MODELİ (Markov Zinciri)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/markov_zinciri.png'
    plt.savefig(dosya, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def dashboard_ciz(model, X_test_olcekli, y_test, gecis_matrisi, kayit_klasoru: str = '.'):
    """
    Gemi Makine Dairesi Erken Uyarı Dashboard'u.
    Sensör Grubu | Risk Skoru | Tahmin | Aksiyon tablosu.
    """
    tahminler   = model.predict(X_test_olcekli)
    olasiliklar = model.predict_proba(X_test_olcekli)
    risk_skorlari = (1 - olasiliklar[:, 0]) * 100
    y_arr = y_test.values

    # Dashboard verisi
    sensor_gruplari = [
        ('Hava Sistemi',   [1], '🔵'),
        ('Yakıt Sistemi',  [0], '🟢'),
        ('Egzoz Sistemi',  [2], '🔴'),
    ]

    fig = plt.figure(figsize=(16, 10))
    gs_main = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Sol üst: Risk dağılımı ──────────────────
    ax1 = fig.add_subplot(gs_main[0, 0])
    for sinif, renk, etiket in [(0, '#2ecc71', 'Normal'),
                                 (1, '#e74c3c', 'Komp.Arıza'),
                                 (2, '#e67e22', 'Türb.Arıza')]:
        maske = y_arr == sinif
        if maske.sum() > 0:
            ax1.hist(risk_skorlari[maske], bins=25, alpha=0.6, color=renk,
                     label=f'{etiket} (n={maske.sum()})', edgecolor='white')
    ax1.axvline(50, color='black', linestyle='--', linewidth=1.5, label='Eşik %50')
    ax1.axvline(75, color='darkred', linestyle=':', linewidth=1.5, label='Kritik %75')
    ax1.set_xlabel('Risk Skoru (%)')
    ax1.set_ylabel('Kayıt Sayısı')
    ax1.set_title('Risk Skoru Dağılımı', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Sağ üst: Markov geçiş (kompresör başlangıcı) ──
    ax2 = fig.add_subplot(gs_main[0, 1])
    adimlar = list(range(1, 9))
    from zincir_analizi import n_adim_ileri_tahmini
    for sinif_idx, (renk, etiket) in enumerate(zip(RENKLER_SINIF, SINIF_ISIMLER)):
        vals = [n_adim_ileri_tahmini(gecis_matrisi, 1, a)[sinif_idx] * 100 for a in adimlar]
        ax2.plot(adimlar, vals, 'o-', color=renk, label=etiket, linewidth=2)
    ax2.set_xlabel('Ölçüm Adımı (t+n)')
    ax2.set_ylabel('Olasılık (%)')
    ax2.set_title('Zincirleme İlerleme\n(Başlangıç: Kompresör Arızası)', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(adimlar)
    ax2.set_xticklabels([f't+{a}' for a in adimlar])

    # ── Alt: Senaryo Tablosu ────────────────────
    ax3 = fig.add_subplot(gs_main[1, :])
    ax3.axis('off')

    # Yüksek riskli 6 örnek
    en_yuksek_idx = np.argsort(risk_skorlari)[::-1][:6]
    sinif_ismi_kisa = {0: 'Normal', 1: 'Komp.Arıza', 2: 'Türb.Arıza'}

    tablo_basliklar = ['#', 'Risk Skoru', 'Tahmin', 'Gerçek', 'Doğru?',
                       'Zincirleme (t+4)', 'Risk Seviyesi', 'Aksiyon']
    tablo_satirlar = []

    for sira, idx in enumerate(en_yuksek_idx, 1):
        tahmin = tahminler[idx]
        gercek = y_arr[idx]
        risk   = risk_skorlari[idx]
        dogru  = '✔' if tahmin == gercek else '✗'

        zincir_4 = n_adim_ileri_tahmini(gecis_matrisi, int(tahmin), n_adim=4)
        zincir_str = (f"N=%{zincir_4[0]*100:.0f} "
                      f"K=%{zincir_4[1]*100:.0f} "
                      f"T=%{zincir_4[2]*100:.0f}")

        if risk >= 75:
            seviye = 'KRİTİK 🚨'
            aksiyon = 'Acil müdahale / Yedek geçiş'
        elif risk >= 50:
            seviye = 'YÜKSEK ⚠'
            aksiyon = 'Sensör takibi / Bakım bildir'
        else:
            seviye = 'NORMAL ✅'
            aksiyon = 'Rutin izleme'

        tablo_satirlar.append([
            str(sira),
            f'%{risk:.1f}',
            sinif_ismi_kisa[int(tahmin)],
            sinif_ismi_kisa[int(gercek)],
            dogru,
            zincir_str,
            seviye,
            aksiyon,
        ])

    tablo = ax3.table(
        cellText=tablo_satirlar,
        colLabels=tablo_basliklar,
        loc='center',
        cellLoc='center'
    )
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(8.5)
    tablo.scale(1, 1.8)

    for (satir, sutun), hucre in tablo.get_celld().items():
        if satir == 0:
            hucre.set_facecolor('#2c3e50')
            hucre.set_text_props(color='white', fontweight='bold')
        elif satir > 0:
            risk_val = float(tablo_satirlar[satir-1][1].replace('%', ''))
            if risk_val >= 75:
                hucre.set_facecolor('#fadbd8')
            elif risk_val >= 50:
                hucre.set_facecolor('#fef9e7')
            else:
                hucre.set_facecolor('#eafaf1')
            if tablo_satirlar[satir-1][4] == '✗':
                if sutun in [2, 3, 4]:
                    hucre.set_facecolor('#f1948a')

    ax3.set_title('GEMİ MAKİNE DAİRESİ — ERKEN UYARI & AKSİYON PLANI TABLOSU',
                  fontsize=12, fontweight='bold', pad=15)

    fig.suptitle('GEMİ ARIZA ZİNCİRİ TAHMİN — DASHBOARD',
                 fontsize=15, fontweight='bold', y=1.01)
    dosya = f'{kayit_klasoru}/dashboard.png'
    plt.savefig(dosya, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Kaydedildi: {dosya}")


def risk_paneli_ciz(model, X_test_olcekli, y_test, kayit_klasoru: str = '.'):
    """Basit risk dağılım paneli (geriye uyumluluk için korundu)."""
    tahminler   = model.predict(X_test_olcekli)
    olasiliklar = model.predict_proba(X_test_olcekli)
    risk_skoru  = (1 - olasiliklar[:, 0]) * 100
    panel = pd.DataFrame({
        'Gerçek Durum': y_test.values,
        'Tahmin':       tahminler,
        'Risk Skoru (%)': risk_skoru.round(1),
    })
    return panel
