"""
=============================================================
MODÜL 3: MODELLEME (GELİŞMİŞ — HİPERPARAMETRE OPTİMİZASYONU)
Ders Notu Algoritmaları:
  - Karar Ağacı   : Gini, overfitting'e yatkın (max_depth kısıtlaması)
  - Rastgele Orman: Bagging ensemble, varyansı düşürür
  - SVM (RBF)     : Marjin maksimizasyonu, doğrusal olmayan sınırlar
Yenilik:
  - GridSearchCV ile otomatik hiperparametre optimizasyonu
  - 5-Fold Stratified Cross-Validation (tek split'e güvenmeme)
=============================================================
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate)


# ─────────────────────────────────────────────
# Çapraz Doğrulama Yardımcısı
# ─────────────────────────────────────────────
def capraz_dogrulama_raporu(model, X, y, model_adi: str) -> dict:
    """
    K-Fold Stratified Cross-Validation uygular.

    Ders Notu Gerekçesi:
        Tek bir %80-%20 bölünmesi şans eseri iyi ya da kötü olabilir.
        5 farklı bölünmede test ederek modelin GERÇEK başarısını ölçeriz.
        Stratified → Her katlama sınıf oranını korur.
        Ortalama ± std dev → Modelin ne kadar kararlı/istikrarlı olduğunu gösterir.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skorlar = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'recall_macro', 'precision_macro', 'f1_macro'],
        n_jobs=-1
    )
    print(f"       5-Fold CV Sonuçları ({model_adi}):")
    print(f"         Doğruluk (Acc) : %{skorlar['test_accuracy'].mean()*100:.2f} ± {skorlar['test_accuracy'].std()*100:.2f}")
    print(f"         Duyarlılık     : %{skorlar['test_recall_macro'].mean()*100:.2f} ± {skorlar['test_recall_macro'].std()*100:.2f}")
    print(f"         F1 Skor        : %{skorlar['test_f1_macro'].mean()*100:.2f} ± {skorlar['test_f1_macro'].std()*100:.2f}")
    return skorlar


# ─────────────────────────────────────────────
# Karar Ağacı + GridSearch
# ─────────────────────────────────────────────
def karar_agaci_egit(X_egitim, y_egitim) -> DecisionTreeClassifier:
    """
    GridSearchCV ile en iyi Karar Ağacı hiperparametrelerini bulur.

    Ders Notu Gerekçesi:
        max_depth=None → ağaç sonuna kadar büyür → OVERFITTING
        GridSearch ile en iyi max_depth ve min_samples_leaf otomatik seçilir.
        criterion='gini' → Gini Impurity minimizasyonu
    """
    print("  [3a] Karar Ağacı — GridSearchCV ile hiperparametre arama...")
    param_grid = {
        'max_depth':        [8, 12],
        'min_samples_leaf': [5, 10, 20],
        'criterion':        ['gini']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid, cv=cv,
        scoring='recall_macro',   # Gemi emniyeti: Recall öncelik
        n_jobs=-1, verbose=0
    )
    gs.fit(X_egitim, y_egitim)
    print(f"       En iyi parametreler : {gs.best_params_}")
    print(f"       En iyi CV Recall    : %{gs.best_score_*100:.2f}")
    model = gs.best_estimator_
    capraz_dogrulama_raporu(model, X_egitim, y_egitim, 'Karar Ağacı')
    return model


# ─────────────────────────────────────────────
# Rastgele Orman + GridSearch
# ─────────────────────────────────────────────
def rastgele_orman_egit(X_egitim, y_egitim) -> RandomForestClassifier:
    """
    GridSearchCV ile Rastgele Orman hiperparametrelerini optimize eder.

    Ders Notu Gerekçesi:
        Bagging (Bootstrap Aggregating) → Her ağaç farklı veri alt kümesi görür.
        n_estimators arttıkça varyans düşer ama işlem süresi artar.
        max_features='sqrt' → Her bölünmede √n özellik → ağaçlar birbirinden bağımsız kalır.
        class_weight='balanced' → Dengesiz sınıflarda azınlık sınıfı ağırlıklandırılır.
    """
    print("  [3b] Rastgele Orman — GridSearchCV ile hiperparametre arama...")
    param_grid = {
        'n_estimators': [100],
        'max_depth':    [12],
        'max_features': ['sqrt']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        param_grid, cv=cv,
        scoring='recall_macro',
        n_jobs=-1, verbose=0
    )
    gs.fit(X_egitim, y_egitim)
    print(f"       En iyi parametreler : {gs.best_params_}")
    print(f"       En iyi CV Recall    : %{gs.best_score_*100:.2f}")
    model = gs.best_estimator_
    capraz_dogrulama_raporu(model, X_egitim, y_egitim, 'Rastgele Orman')
    return model


# ─────────────────────────────────────────────
# SVM + GridSearch
# ─────────────────────────────────────────────
def svm_egit(X_egitim, y_egitim) -> SVC:
    """
    GridSearchCV ile SVM C ve gamma parametrelerini optimize eder.

    Ders Notu Gerekçesi:
        C (Regularization): Büyük C → marjin daralır, az hata tolere eder
                            Küçük C → geniş marjin, daha fazla hata kabul
        RBF Kernel: Veriyi yüksek boyutlu uzaya taşıyarak doğrusal olmayan
                    arıza sınırlarını çizer (Kernel Trick).
        gamma='scale' → 1/(n_özellik × X.var()) — otomatik ölçekleme
    """
    print("  [3c] SVM (RBF Kernel) — GridSearchCV ile hiperparametre arama...")
    param_grid = {
        'C':     [10],
        'gamma': ['scale']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        SVC(kernel='rbf', probability=True,
            class_weight='balanced',
            decision_function_shape='ovr',
            random_state=42),
        param_grid, cv=cv,
        scoring='recall_macro',
        n_jobs=-1, verbose=0
    )
    gs.fit(X_egitim, y_egitim)
    print(f"       En iyi parametreler : {gs.best_params_}")
    print(f"       En iyi CV Recall    : %{gs.best_score_*100:.2f}")
    model = gs.best_estimator_
    capraz_dogrulama_raporu(model, X_egitim, y_egitim, 'SVM (RBF)')
    return model


# ─────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────
def modelleri_egit(X_egitim, y_egitim) -> dict:
    """Tüm modelleri GridSearchCV ile eğitir."""
    print("=" * 60)
    print("  [3/5] MODELLEME + HİPERPARAMETRE OPTİMİZASYONU")
    print("=" * 60)
    print("  (GridSearchCV + 5-Fold Stratified CV — bu adım birkaç dakika sürer)")
    print()

    modeller = {
        'Karar Ağacı':    karar_agaci_egit(X_egitim, y_egitim),
        'Rastgele Orman': rastgele_orman_egit(X_egitim, y_egitim),
        'SVM (RBF)':      svm_egit(X_egitim, y_egitim),
    }
    print()
    print("  ✔ Tüm modeller optimize edildi ve eğitildi.")
    return modeller
