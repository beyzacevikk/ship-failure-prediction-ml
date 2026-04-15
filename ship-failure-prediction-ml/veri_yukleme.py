"""
=============================================================
MODÜL 1: VERİ YÜKLEME
Gemi sensör verisini okur, temel istatistikleri raporlar.
=============================================================
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------
# SABİTLER
# -----------------------------------------------------------
SUTUN_ADLARI = [
    'kol_konumu', 'gemi_hizi', 'gt_saft_torku', 'gt_devir',
    'gaz_jeneratoru_devri', 'sancak_pervane_torku', 'iskele_pervane_torku',
    'hp_turbin_cikis_sicakligi', 'komp_giris_sicaklik', 'komp_cikis_sicaklik',
    'hp_turbin_cikis_basinci', 'komp_giris_basinc', 'komp_cikis_basinc',
    'egzoz_gaz_basinci', 'turbin_enjeksiyon_kontrolu', 'yakit_akisi',
    'kompresor_asinma_katsayisi', 'turbin_asinma_katsayisi'
]


def veri_yukle(dosya_yolu: str) -> pd.DataFrame:
    """
    CSV dosyasını yükler ve temel kalite kontrolü yapar.

    Parametre:
        dosya_yolu: Naval Plant dataset yolu

    Dönüş:
        Ham DataFrame
    """
    print("=" * 60)
    print("  [1/5] VERİ YÜKLEME")
    print("=" * 60)

    veri = pd.read_csv(
        dosya_yolu,
        sep=r'\s+',
        header=None,
        names=SUTUN_ADLARI
    )

    print(f"  ✔ Toplam kayıt sayısı : {len(veri):,}")
    print(f"  ✔ Özellik sayısı      : {veri.shape[1]}")
    print(f"  ✔ Eksik değer sayısı  : {veri.isnull().sum().sum()}")
    print()

    # Hedef değişken dağılımını göster
    print("  Aşınma katsayısı aralıkları:")
    print(f"    Kompresör  → min: {veri['kompresor_asinma_katsayisi'].min():.3f}  "
          f"max: {veri['kompresor_asinma_katsayisi'].max():.3f}  "
          f"ort: {veri['kompresor_asinma_katsayisi'].mean():.3f}")
    print(f"    Türbin     → min: {veri['turbin_asinma_katsayisi'].min():.3f}  "
          f"max: {veri['turbin_asinma_katsayisi'].max():.3f}  "
          f"ort: {veri['turbin_asinma_katsayisi'].mean():.3f}")

    return veri
