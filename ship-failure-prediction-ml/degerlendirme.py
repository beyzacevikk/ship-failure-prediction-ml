"""
=============================================================
MODÜL 4: MODEL DEĞERLENDİRME (GELİŞMİŞ)
  - Accuracy, Recall, Precision, F1
  - Confusion Matrix
  - False Negative Emniyet Analizi (Kritik Bölüm)
  - Özellik önem sıralaması
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

SINIF_ISIMLER = {0: 'Normal', 1: 'Kompresör Arızası', 2: 'Türbin Arızası'}


def model_degerlendir(model, X_test, y_test, model_adi: str) -> dict:
    tahminler = model.predict(X_test)
    return {
        'model_adi':          model_adi,
        'dogruluk':           accuracy_score(y_test, tahminler),
        'duyarlilik':         recall_score(y_test, tahminler, average='macro', zero_division=0),
        'kesinlik':           precision_score(y_test, tahminler, average='macro', zero_division=0),
        'f1_skor':            f1_score(y_test, tahminler, average='macro', zero_division=0),
        'karisiklik_matrisi': confusion_matrix(y_test, tahminler),
        'tahminler':          tahminler,
        'y_test':             y_test,
    }


def sonuclari_yazdir(tum_metrikler: list):
    print("=" * 60)
    print("  [4/5] MODEL DEĞERLENDİRME")
    print("=" * 60)
    print()
    print("  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Model               │ Doğruluk │Duyarlılık│ Kesinlik │ F1 Skor  │")
    print("  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    for m in tum_metrikler:
        print(f"  │ {m['model_adi']:<19} │  {m['dogruluk']*100:>5.2f}%  │"
              f"  {m['duyarlilik']*100:>5.2f}%  │  {m['kesinlik']*100:>5.2f}%  │"
              f"  {m['f1_skor']*100:>5.2f}%  │")
    print("  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘")


def en_iyi_modeli_sec(tum_metrikler: list) -> dict:
    en_iyi = max(tum_metrikler, key=lambda x: x['duyarlilik'])
    print(f"\n  ► EN İYİ MODEL (Recall bazında): {en_iyi['model_adi']}")
    print(f"    Duyarlılık : %{en_iyi['duyarlilik']*100:.2f}")
    print(f"    Doğruluk   : %{en_iyi['dogruluk']*100:.2f}")
    return en_iyi


def false_negative_emniyet_analizi(tum_metrikler: list):
    """
    ★ EMNİYET KRİTİK ANALİZ: False Negative Risk Raporu

    Ders Notu Gerekçesi:
        Deniz emniyetinde False Negative (arızayı gözden kaçırma) en büyük risktir.
        "Arıza VAR ama model NORMAL dedi" → Gemi açık denizde mahsur kalır.

        False Negative Oranı = FN / (FN + TP)  [= 1 - Recall]
        Her sınıf için bu oran ayrı ayrı hesaplanır ve maliyet yorumu yapılır.
    """
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   EMNİYET KRİTİK ANALİZ: FALSE NEGATIVE RİSKİ      ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    for metrik in tum_metrikler:
        cm = metrik['karisiklik_matrisi']
        print(f"  ▌ {metrik['model_adi']}")

        for sinif_idx in range(cm.shape[0]):
            TP = cm[sinif_idx, sinif_idx]
            FN = cm[sinif_idx, :].sum() - TP
            toplam_gercek = cm[sinif_idx, :].sum()
            if toplam_gercek == 0:
                continue
            fn_orani = FN / toplam_gercek * 100
            recall_sinif = TP / toplam_gercek * 100
            sinif_adi = SINIF_ISIMLER[sinif_idx]

            sembol = "✔" if fn_orani < 5 else ("⚠" if fn_orani < 15 else "🚨")
            print(f"    {sembol} {sinif_adi:<22}: "
                  f"FN={FN:4d} | FN Oranı=%{fn_orani:5.1f} | Recall=%{recall_sinif:5.1f}")

            # Maliyet yorumu
            if sinif_idx == 1 and fn_orani > 10:
                print(f"         ⛔ YORUM: Kompresör arızalarının %{fn_orani:.1f}'i gözden kaçıyor!")
                print(f"            → Bu kayıtlarda sistem 'Normal' dedi, ama arıza VAR.")
            elif sinif_idx == 2 and fn_orani > 10:
                print(f"         ⛔ YORUM: Türbin arızalarının %{fn_orani:.1f}'i gözden kaçıyor!")
                print(f"            → Türbin arızası zincirleme hasar üretebilir.")
        print()

    print("  ★ Mühendislik Notu:")
    print("    Makine öğrenmesi projemizde kurallar önceden yazılmadığı için")
    print("    sistem, veriler arasındaki matematiksel ilişkileri kullanarak")
    print("    erken uyarı üretir. Recall yüksek = az FN = güvenli deniz yolculuğu.")


def ozellik_onemliligi_yazdir(model, ozellik_isimleri: list):
    if not hasattr(model, 'feature_importances_'):
        return
    print("\n  Özellik Önem Sıralaması (Rastgele Orman — Top 10):")
    onemler = model.feature_importances_
    siralama = np.argsort(onemler)[::-1]
    for i, idx in enumerate(siralama[:10], 1):
        bar = "█" * int(onemler[idx] * 100)
        print(f"    {i:2}. {ozellik_isimleri[idx]:<38} {onemler[idx]*100:5.2f}%  {bar}")
