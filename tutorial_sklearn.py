"""
tutorial_sklearn.py: cinque mini-esperimenti per capire gli strumenti di sklearn usati nell'EDA.

    python tutorial_sklearn.py

Ogni sezione stampa un risultato che dovresti saper prevedere PRIMA di eseguirla.
I dati sono finti (numpy): l'obiettivo è il meccanismo, non i numeri.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)


def titolo(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ---------------------------------------------------------------------------
titolo("1. L'API di sklearn: fit / transform / predict, e gli attributi con il trattino basso")
X = rng.normal(loc=[10, 0], scale=[5, 0.1], size=(200, 2))     # due colonne con scale molto diverse
scaler = StandardScaler()
scaler.fit(X)                                  # IMPARA (media e deviazione standard) dai dati
Z = scaler.transform(X)                        # APPLICA quanto imparato
print("media imparata      :", scaler.mean_.round(2))          # attributi imparati finiscono con "_"
print("dopo transform, media:", Z.mean(0).round(2), " std:", Z.std(0).round(2))
print("=> fit e transform sono due passi distinti: è ciò che permette il punto 2.")

# ---------------------------------------------------------------------------
titolo("2. Lo scaler va adattato SOLO al riferimento (knn.py, knn_scores)")
riferimento = rng.normal(0, 1, size=(100, 3))
test_ok = rng.normal(0, 1, size=(20, 3))
test_anomalo = rng.normal(0, 1, size=(20, 3)) + [8, 0, 0]       # spostato sulla prima colonna
# Modo giusto: lo scaler impara dal solo riferimento
s_ok = StandardScaler().fit(riferimento)
# Modo sbagliato: lo scaler vede anche i test anomali, quindi ne "assorbe" la scala
s_male = StandardScaler().fit(np.vstack([riferimento, test_ok, test_anomalo]))
for nome, s in (("giusto ", s_ok), ("sbagliato", s_male)):
    nn = NearestNeighbors(n_neighbors=1).fit(s.transform(riferimento))
    d_ok = nn.kneighbors(s.transform(test_ok))[0].mean()
    d_an = nn.kneighbors(s.transform(test_anomalo))[0].mean()
    print(f"{nome}: distanza media normali={d_ok:.2f}  anomali={d_an:.2f}  rapporto={d_an / d_ok:.2f}")
print("=> con lo scaler sbagliato il rapporto si comprime: il test ha influenzato la scala.")

# ---------------------------------------------------------------------------
titolo("3. PCA: quanta varianza tengo? (domain_shift.py, plot_pca)")
base = rng.normal(size=(300, 2))
X = np.hstack([base, base @ rng.normal(size=(2, 8)) + 0.05 * rng.normal(size=(300, 8))])   # 10 colonne, 2 vere
pca = PCA(n_components=5).fit(StandardScaler().fit_transform(X))
print("varianza spiegata per componente:", pca.explained_variance_ratio_.round(3))
print("=> due componenti bastano: le altre 8 colonne sono combinazioni delle prime due.")

# ---------------------------------------------------------------------------
titolo("4. Perché servono i gruppi nella validazione incrociata (domain_shift.py, _cv_auc)")
# 20 "rulli", ciascuno con 10 patch quasi identiche. L'etichetta (train/test) è decisa PER RULLO
# a caso: non esiste nessuno shift reale, quindi l'AUC onesta è 0.5.
n_gruppi, per_gruppo, d = 20, 10, 30
centri = rng.normal(size=(n_gruppi, d))
X = np.repeat(centri, per_gruppo, axis=0) + 0.05 * rng.normal(size=(n_gruppi * per_gruppo, d))
gruppo = np.repeat(np.arange(n_gruppi), per_gruppo)
y_gruppo = np.array([0, 1] * (n_gruppi // 2))
rng.shuffle(y_gruppo)
y = np.repeat(y_gruppo, per_gruppo)
modello = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))

cv_semplice = StratifiedKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(modello, X, y, cv=cv_semplice, method="decision_function")
print(f"StratifiedKFold        (ignora i gruppi): AUC = {roc_auc_score(y, p):.2f}")
cv_gruppi = StratifiedGroupKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(modello, X, y, cv=cv_gruppi, groups=gruppo, method="decision_function")
print(f"StratifiedGroupKFold   (rispetta i gruppi): AUC = {roc_auc_score(y, p):.2f}")
print("=> senza i gruppi il modello 'riconosce' il rullo dalle patch vicine e l'AUC è gonfiata.")
print("   (con soli 20 rulli l'AUC onesta oscilla attorno a 0.5: un valore come 0.35 è rumore.)")

# ---------------------------------------------------------------------------
titolo("5. Metriche: AUROC e AP, e cosa vale il caso (knn.py, evaluate)")
n_neg, n_pos = 90, 10
punteggi = np.r_[rng.normal(0, 1, n_neg), rng.normal(1.5, 1, n_pos)]
etichette = np.r_[np.zeros(n_neg), np.ones(n_pos)]
print(f"AUROC = {roc_auc_score(etichette, punteggi):.2f}   (caso = 0.50, indipendente dalla prevalenza)")
print(f"AP    = {average_precision_score(etichette, punteggi):.2f}   (caso = prevalenza = {n_pos / (n_pos + n_neg):.2f})")
print("=> con pochi positivi l'AP va sempre letta rispetto alla prevalenza (colonna ap_chance).")
print("   Inoltre roc_auc_score vuole PUNTEGGI continui, non le etichette predette (predict).")
