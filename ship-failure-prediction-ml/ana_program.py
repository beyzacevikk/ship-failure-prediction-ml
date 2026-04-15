"""
=============================================================
GEMİ ARIZA ZİNCİRİ TAHMİN VE RİSK ANALİZİ SİSTEMİ v3.0
Failure Propagation Prediction System

  1. Problem Tanımı  -> Classification + Regression
  2. Veri Yukleme    -> veri_yukleme.py
  3. EDA             -> eda.py (YENİ)
  4. Veri Temizleme  -> on_isleme.py
  5. Regresyon       -> regresyon.py (YENİ)
  6. Classification  -> modelleme.py
  7. Degerlendirme   -> degerlendirme.py
  8. Gorsellestirme  -> gorsellestirme.py
  9. Zincir Analizi  -> zincir_analizi.py
=============================================================
"""

import os, sys
PROJE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJE)

from veri_yukleme   import veri_yukle
from eda            import eda_rapor
from on_isleme      import on_isle
from regresyon      import regresyon_analizi_yap
from modelleme      import modelleri_egit
from degerlendirme  import (model_degerlendir, sonuclari_yazdir,
                             en_iyi_modeli_sec, false_negative_emniyet_analizi,
                             ozellik_onemliligi_yazdir)
from gorsellestirme import (karisiklik_matrisi_ciz, model_karsilastirma_ciz,
                             ozellik_onemliligi_ciz, markov_matrisi_ciz,
                             dashboard_ciz)
from zincir_analizi import zincir_analizi_yap


def ana_program():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  GEMİ ARIZA ZİNCİRİ TAHMİN VE RİSK ANALİZİ SİSTEMİ    ║")
    print("║  Failure Propagation Prediction System  v3.0             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    cikti = PROJE

    # 1. VERİ YÜKLEME
    veri_dosyasi = os.path.join(PROJE, 'navalplantmaintenance.csv')
    ham_veri = veri_yukle(veri_dosyasi)
    print()

    # 2. EDA
    eda_rapor(ham_veri, cikti)
    print()

    # 3. REGRESYON ANALİZİ
    regresyon_sonuclari = regresyon_analizi_yap(ham_veri, cikti)
    print()

    # 4. VERİ ÖN İŞLEME
    (X_egitim, X_test, y_egitim, y_test,
     olceklendirici, ozellik_isimleri) = on_isle(ham_veri)
    print()

    # 5. CLASSIFICATION
    modeller = modelleri_egit(X_egitim, y_egitim)
    print()

    # 6. DEĞERLENDİRME
    tum_metrikler = [model_degerlendir(m, X_test, y_test, ad)
                     for ad, m in modeller.items()]
    sonuclari_yazdir(tum_metrikler)
    en_iyi = en_iyi_modeli_sec(tum_metrikler)
    ozellik_onemliligi_yazdir(modeller['Rastgele Orman'], ozellik_isimleri)
    false_negative_emniyet_analizi(tum_metrikler)
    print()

    # 7. GÖRSEL
    print("=" * 60)
    print("  [VIZ] GÖRSELLEŞTİRME")
    print("=" * 60)
    karisiklik_matrisi_ciz(tum_metrikler, cikti)
    model_karsilastirma_ciz(tum_metrikler, cikti)
    ozellik_onemliligi_ciz(modeller['Rastgele Orman'], ozellik_isimleri, cikti)

    # 8. ZİNCİR
    en_iyi_model = modeller[en_iyi['model_adi']]
    zincir_sonuclari = zincir_analizi_yap(
        en_iyi_model, X_test, y_test, y_egitim, ozellik_isimleri
    )
    markov_matrisi_ciz(zincir_sonuclari['gecis_matrisi'], cikti)
    dashboard_ciz(en_iyi_model, X_test, y_test,
                  zincir_sonuclari['gecis_matrisi'], cikti)

    # ÖZET
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SİSTEM HAZIR                                            ║")
    print(f"║  En İyi Sinif. Model: {en_iyi['model_adi']:<35}║")
    print(f"║  Recall (Duyarlilik): %{en_iyi['duyarlilik']*100:.2f}{' '*33}║")
    print(f"║  Accuracy (Dogruluk): %{en_iyi['dogruluk']*100:.2f}{' '*33}║")
    en_iyi_komp_ad = regresyon_sonuclari['metrik_komp']['__en_iyi__']
    en_iyi_turb_ad = regresyon_sonuclari['metrik_turb']['__en_iyi__']
    r2_komp = regresyon_sonuclari['metrik_komp'][en_iyi_komp_ad]['r2']
    r2_turb = regresyon_sonuclari['metrik_turb'][en_iyi_turb_ad]['r2']
    print(f"║  Regresyon Komp. R2 : {r2_komp:.4f}{' '*33}║")
    print(f"║  Regresyon Turb. R2 : {r2_turb:.4f}{' '*33}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    return modeller, tum_metrikler, zincir_sonuclari, olceklendirici, ozellik_isimleri


if __name__ == '__main__':
    modeller, metrikler, zincir, olcek, ozellikler = ana_program()
