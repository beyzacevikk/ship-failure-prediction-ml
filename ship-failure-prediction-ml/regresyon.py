"""
=============================================================
MODÜL: REGRESYON ANALİZİ
Pipeline Adım 3 — Sürekli Hedef Tahmin

Aşınma katsayısını SAYISAL olarak tahmin eder (0-1 arasi deger).
  Classification -> "Ariza var mi?" sorusunu yanitlar.
  Regression     -> "Asınma ne kadar?" sorusunu yanitlar.

Modeller:
  - Linear Regression (Ridge) : Temel referans modeli
  - Random Forest Regressor   : Ensemble non-linear regresyon
  - Gradient Boosting Reg.    : Gradient Boosting (en guclu)

Metrikler:
  - MAE  : Ortalama Mutlak Hata
  - RMSE : Karekök Ortalama Kare Hata
  - R2   : Belirleme Katsayisi (ne kadar acikliyor?)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def regresyon_verisi_hazirla(ham_veri: pd.DataFrame) -> tuple:
    """
    Regresyon icin X (ozellikler) ve y (asinma katsayisi) ayirir.
    Sabit sutunlar cikarilir, olcekleme uygulanir.
    """
    sabit = ['komp_giris_sicaklik', 'komp_giris_basinc']
    temiz = ham_veri.drop(columns=sabit, errors='ignore').copy()

    y_komp = temiz['kompresor_asinma_katsayisi']
    y_turb = temiz['turbin_asinma_katsayisi']
    X = temiz.drop(columns=['kompresor_asinma_katsayisi', 'turbin_asinma_katsayisi'])

    # Basit ozellik muhendisligi
    for col in ['komp_cikis_basinc', 'komp_cikis_sicaklik', 'yakit_akisi', 'gt_devir']:
        if col in X.columns:
            X[f'{col}_fark1'] = X[col].diff().fillna(0)

    X_train, X_test, yk_train, yk_test, yt_train, yt_test = train_test_split(
        X, y_komp, y_turb, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return (X_train_sc, X_test_sc, yk_train, yk_test, yt_train, yt_test,
            scaler, X.columns.tolist())


def regresyon_modelleri_egit(X_train, y_train, hedef_adi: str) -> dict:
    """3 regresyon modeli egitir, CV skoru hesaplar (sadece Ridge icin hizli CV)."""
    modeller_tanim = {
        'Linear Reg. (Ridge)': Ridge(alpha=1.0),
        'Random Forest Reg.' : RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        ),
        'Gradient Boosting'  : GradientBoostingRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
    }

    sonuclar = {}
    print(f"\n  Hedef: {hedef_adi}")
    print(f"  {'Model':<25} {'CV R² (3-fold)':>16} {'CV RMSE':>12}")
    print("  " + "-" * 56)

    for ad, model in modeller_tanim.items():
        cv_r2   = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
        cv_rmse = cross_val_score(model, X_train, y_train, cv=3,
                                  scoring='neg_root_mean_squared_error', n_jobs=-1)
        model.fit(X_train, y_train)
        sonuclar[ad] = {
            'model'      : model,
            'cv_r2_ort'  : cv_r2.mean(),
            'cv_r2_std'  : cv_r2.std(),
            'cv_rmse_ort': -cv_rmse.mean(),
        }
        print(f"  {ad:<25} {cv_r2.mean():>+.4f} +/- {cv_r2.std():.4f}  "
              f"{-cv_rmse.mean():>10.5f}")

    return sonuclar


def regresyon_degerlendir(sonuclar: dict, X_test, y_test, hedef_adi: str) -> dict:
    """Test setinde MAE, RMSE, R2 hesaplar."""
    print(f"\n  +---------------------------+----------+----------+----------+")
    print(f"  | Model ({hedef_adi[:12]:<12})     |   MAE    |   RMSE   |    R2    |")
    print(f"  +---------------------------+----------+----------+----------+")

    en_iyi_r2 = -np.inf
    en_iyi_ad = None
    metrikler = {}

    for ad, bilgi in sonuclar.items():
        tahmin = bilgi['model'].predict(X_test)
        mae    = mean_absolute_error(y_test, tahmin)
        rmse   = np.sqrt(mean_squared_error(y_test, tahmin))
        r2     = r2_score(y_test, tahmin)
        metrikler[ad] = {'tahmin': tahmin, 'mae': mae, 'rmse': rmse, 'r2': r2}
        print(f"  | {ad:<26}| {mae:.5f}  | {rmse:.5f}  | {r2:+.4f}  |")
        if r2 > en_iyi_r2:
            en_iyi_r2, en_iyi_ad = r2, ad

    print(f"  +---------------------------+----------+----------+----------+")
    print(f"  >> En iyi: {en_iyi_ad}  (R2={en_iyi_r2:.4f})")
    metrikler['__en_iyi__'] = en_iyi_ad
    return metrikler


def regresyon_grafik_ciz(metrik_komp, metrik_turb, yk_test, yt_test,
                          kayit_klasoru: str = '.'):
    """Gercek vs Tahmin scatter + Hata dagilimi."""
    fig, eksenler = plt.subplots(2, 2, figsize=(14, 10))
    hedefler = [
        ('Kompresor Asinma', metrik_komp, yk_test, '#e74c3c'),
        ('Turbin Asinma',    metrik_turb, yt_test, '#e67e22'),
    ]

    for satir, (hedef_adi, metrikler, y_test, renk) in enumerate(hedefler):
        en_iyi_ad = metrikler['__en_iyi__']
        m         = metrikler[en_iyi_ad]
        tahmin    = m['tahmin']
        y_arr     = np.array(y_test)

        # Sol: Gercek vs Tahmin
        ax1 = eksenler[satir, 0]
        ax1.scatter(y_arr, tahmin, alpha=0.3, s=8, color=renk)
        lim = [min(y_arr.min(), tahmin.min()) - 0.002,
               max(y_arr.max(), tahmin.max()) + 0.002]
        ax1.plot(lim, lim, 'k--', linewidth=1.5, label='Mukemmel tahmin')
        ax1.set_xlabel('Gercek Deger', fontsize=10)
        ax1.set_ylabel('Tahmin Edilen', fontsize=10)
        ax1.set_title(f'{hedef_adi} — Gercek vs Tahmin\n'
                      f'{en_iyi_ad}  |  R2={m["r2"]:.4f}  RMSE={m["rmse"]:.5f}',
                      fontsize=9, fontweight='bold')
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        # Sag: Hata dagilimi
        ax2 = eksenler[satir, 1]
        hatalar = tahmin - y_arr
        ax2.hist(hatalar, bins=50, color=renk, alpha=0.75, edgecolor='white')
        ax2.axvline(0,              color='black', linestyle='--', lw=1.5, label='Sifir Hata')
        ax2.axvline(hatalar.mean(), color='blue',  linestyle=':',  lw=1.5,
                    label=f'Ort. Hata: {hatalar.mean():.5f}')
        ax2.set_xlabel('Hata (Tahmin - Gercek)', fontsize=10)
        ax2.set_ylabel('Frekans', fontsize=10)
        ax2.set_title(f'{hedef_adi} — Hata Dagilimi\nMAE={m["mae"]:.5f}',
                      fontsize=9, fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle('REGRESYON ANALIZI — Asinma Katsayisi Tahmini\n'
                 '(0 = kritik asinma, 1 = yeni)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/regresyon_sonuclari.png'
    plt.savefig(dosya, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {dosya}")


def regresyon_karsilastirma_grafigi(metrik_komp, metrik_turb,
                                    kayit_klasoru: str = '.'):
    """Tum modellerin R2 ve RMSE karsilastirmasi."""
    model_adlari = [k for k in metrik_komp if not k.startswith('__')]
    r2_k  = [metrik_komp[k]['r2']   for k in model_adlari]
    r2_t  = [metrik_turb[k]['r2']   for k in model_adlari]
    rmse_k = [metrik_komp[k]['rmse'] for k in model_adlari]
    rmse_t = [metrik_turb[k]['rmse'] for k in model_adlari]

    x = np.arange(len(model_adlari)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vals_k, vals_t, ylabel, title in [
        (ax1, r2_k,   r2_t,   'R2 Skoru',  'Belirleme Katsayisi (R2)'),
        (ax2, rmse_k, rmse_t, 'RMSE',      'RMSE — Dusuk = Iyi'),
    ]:
        b1 = ax.bar(x - w/2, vals_k, w, label='Kompresor', color='#e74c3c', alpha=0.85)
        b2 = ax.bar(x + w/2, vals_t, w, label='Turbin',    color='#e67e22', alpha=0.85)
        for bars in [b1, b2]:
            for bar in bars:
                ax.annotate(f'{bar.get_height():.4f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace(' ', '\n') for a in model_adlari], fontsize=8)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Regresyon Modeli Karsilastirmasi', fontsize=12, fontweight='bold')
    plt.tight_layout()
    dosya = f'{kayit_klasoru}/regresyon_karsilastirma.png'
    plt.savefig(dosya, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {dosya}")


def regresyon_analizi_yap(ham_veri: pd.DataFrame, kayit_klasoru: str = '.') -> dict:
    """Tam regresyon pipeline'i."""
    print("=" * 60)
    print("  [REG] REGRESYON ANALIZI — ASINMA KATSAYISI TAHMINI")
    print("=" * 60)
    print()

    (X_train, X_test, yk_train, yk_test,
     yt_train, yt_test, scaler, _) = regresyon_verisi_hazirla(ham_veri)

    print("  Kompresor Asinma Katsayisi Regresyonu:")
    sonuc_k = regresyon_modelleri_egit(X_train, yk_train, 'Kompresor Asinma')
    metrik_k = regresyon_degerlendir(sonuc_k, X_test, yk_test, 'Kompresor')

    print("\n  Turbin Asinma Katsayisi Regresyonu:")
    sonuc_t = regresyon_modelleri_egit(X_train, yt_train, 'Turbin Asinma')
    metrik_t = regresyon_degerlendir(sonuc_t, X_test, yt_test, 'Turbin')

    print()
    regresyon_grafik_ciz(metrik_k, metrik_t, yk_test, yt_test, kayit_klasoru)
    regresyon_karsilastirma_grafigi(metrik_k, metrik_t, kayit_klasoru)

    print("\n  Regresyon analizi tamamlandi.")
    return {
        'sonuc_komp'  : sonuc_k,
        'sonuc_turb'  : sonuc_t,
        'metrik_komp' : metrik_k,
        'metrik_turb' : metrik_t,
    }
