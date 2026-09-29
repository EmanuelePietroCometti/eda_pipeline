# EDA del dataset: implementazione di riferimento e guida a sklearn

Questo pacchetto implementa `design.md` (versioni V0/V1/V2, difetti anche nel train,
polvere identificata per differenza tra versioni). **È un riferimento da leggere,
eseguire e modificare**, non un prodotto finito: le soglie e le scelte segnate
[DA CONFERMARE] in `design.md` restano tue.

## 1. Come usarlo

```bash
pip install -r requirements.txt
python tutorial_sklearn.py                     # 5 mini-esperimenti, 10 secondi (vedi §3)

# prova su dati finti con verità nota
python tools/make_synthetic.py --out /tmp/synth --split-mode leaky --rolls 10
#   poi in config.yaml: dataset.versions.V0/V1/V2.path -> /tmp/synth/V0 ... e general.out_dir
python run.py                                  # tutto
python run.py --steps integrity masks          # solo alcuni passi
```

Sul dataset vero: metti i tre percorsi in `config.yaml`, controlla `mask_suffix` e
`gallery.defect_class`, e lancia **prima** `--steps integrity`. Se un controllo
fallisce, guarda `results/<versione>/version_checks.csv` e `mask_issues.csv` prima di
fidarti del resto.

## 2. Mappa: file ↔ analisi

| File | Analisi (design.md) | Cosa fa |
|---|---|---|
| `eda/dataset.py` | §3 e polvere | indice, hash del contenuto, polvere per differenza da V0, controlli di coerenza |
| `eda/masks.py` | 4.1 | componenti connesse, area px e %, frazione sotto una cella |
| `eda/leakage.py` | 4.3 | bordi esatti (hash), pHash, gruppi, quota cross-split osservata e attesa |
| `eda/domain_shift.py` | 4.2 | Cohen's d con bootstrap, classificatore di dominio con nullo per permutazione |
| `eda/knn.py` | 4.4 | distanza kNN dal riferimento, AUROC/AP con IC bootstrap |
| `eda/gallery.py` | 4.5 | campione casuale a seme fisso, con contorno della maschera |
| `eda/features.py` | tutte | lettura delle immagini, embedding `simple` o `resnet18` |
| `run.py` | tutte | ordine dei passi, ripetizione per versione, tabella dei contrasti |

Ordine di lettura consigliato: `dataset.py` → `masks.py` → `leakage.py` →
`domain_shift.py` → `knn.py`. Ogni file si legge da solo; i commenti spiegano il perché.

## 3. Lezioni di sklearn (con dove compaiono nel codice)

`tutorial_sklearn.py` dimostra ciascuna con un esperimento; sotto il concetto.

**L1. L'API: `fit`, `transform`, `predict`.** Ogni oggetto sklearn ha `fit(X)` che
*impara* dai dati e salva il risultato in attributi con il trattino basso finale
(`scaler.mean_`, `pca.components_`), poi `transform(X)` o `predict(X)` che *applicano*.
Tutto il resto discende dal tenere separati i due passi.

**L2. Adatta agli oggetti di riferimento, applica al resto.** In `knn.py`
(`knn_scores`) lo `StandardScaler` è adattato solo a `train/good`. Se lo adatti anche al
test, il test influenza la scala e l'AUROC risulta falsata. Regola generale: qualunque
oggetto che "impara" (scaler, PCA, modello) vede solo dati di addestramento.

**L3. `PCA`.** `PCA(n_components=k).fit(X)`; `explained_variance_ratio_` dice quanta
varianza cattura ogni componente. In `domain_shift.py` (`plot_pca`) serve a vedere in 2D se i
gruppi si separano; nel classificatore riduce le dimensioni prima della regressione.
Va sempre preceduta da `StandardScaler`: senza, comandano le colonne con scala maggiore.

**L4. `LogisticRegression` e i punteggi.** Il classificatore di dominio impara a
distinguere due gruppi. `class_weight="balanced"` compensa gruppi di numerosità diversa.
Per l'AUC si usano punteggi continui (`decision_function`), non le etichette
predette (`predict`): l'AUC valuta l'*ordinamento* dei punteggi.

**L5. `Pipeline` (`make_pipeline`).** Concatena scaler → PCA → classificatore in un solo
oggetto. Dentro la validazione incrociata ogni fold riadatta *tutti* i passi solo sui suoi dati di
addestramento. Senza pipeline è facile adattare lo scaler su tutto il dataset e
contaminare i fold di test senza accorgersene.

**L6. Validazione incrociata con gruppi.** Le patch adiacenti sono quasi identiche:
se finiscono in fold diversi, il modello le "riconosce" e l'AUC risulta gonfiata
(`tutorial_sklearn.py`, sezione 4: 1.00 contro ~0.5). `StratifiedGroupKFold` tiene un
gruppo intero nello stesso fold; i gruppi vengono da `leakage.py` (componenti connesse
delle coppie di bordo). `cross_val_predict` restituisce il punteggio di ogni immagine
quando era nel fold di test, così l'AUC si calcola una volta sola.

**L7. Metriche.** `roc_auc_score` (caso = 0.5, indipendente dalla prevalenza),
`average_precision_score` (il caso vale la prevalenza: per questo `ap_chance` accanto
all'AP), `roc_curve` per il grafico. Con pochi positivi l'intervallo bootstrap
(`knn.py`, `auroc_ci`) è più informativo della stima puntuale.

**L8. `NearestNeighbors`.** `.fit(riferimento)` poi `.kneighbors(X)` restituisce
distanze e indici. Per le immagini che stanno *nel* riferimento il vicino più prossimo
è sé stesse: si scarta (leave-one-out, `knn_scores`).

**L9. Il nullo per permutazione.** `domain_classifier` ripete la validazione con le
etichette mescolate: la distribuzione di AUC che ottieni quando *non* c'è alcun
segnale. L'AUC vera va confrontata con `null_p95`. (sklearn ha `permutation_test_score`,
ma farlo a mano una volta ti mostra cosa fa.)

## 4. Esercizi (in ordine)

1. In `knn_scores` sposta il `fit` dello scaler su tutte le immagini e confronta l'AUROC
   con quella originale. Cosa cambia, e perché?
2. In `domain_shift._cv_auc` togli i gruppi (usa sempre `StratifiedKFold`) e lancia su
   `synth` con `--split-mode clean`. Confronta l'AUC con e senza gruppi.
3. Aggiungi a `masks.py` una colonna con il rapporto `bbox_h / bbox_w` e un grafico:
   i difetti sono allungati?
4. In `leakage.py` abbassa `min_strip_std` a 0 e guarda cosa succede alle coppie: cosa
   corrispondono le strisce uniformi?
5. Cambia `knn.backbone` in `resnet18` (serve torch) e confronta gli AUROC con `simple`.

## 5. Errori tipici (già incontrati)

- **Immagini con pixel identici (duplicati).** Il confronto tra versioni per contenuto conta
  le righe in modo asimmetrico se un contenuto compare due volte: "rimosse = aggiunte" falliva
  per 1 immagine in test (V1) e per 6 in train (V2) sul dataset vero. Ora il controllo confronta
  multinsiemi e `duplicates.csv` elenca i duplicati; le immagini di test identiche a una del
  riferimento hanno distanza kNN 0, quindi l'AUROC si calcola anche senza di esse (colonna `subset`).

- **pandas 3 e i valori mancanti.** In una colonna di stringhe un mancante diventa `NaN`,
  non `None`: `x is None` è sempre falso. Il codice usa `""` e una colonna booleana.
- **Confronti per nome file tra versioni.** Se i nomi cambiano tra versioni, la differenza
  è sbagliata: `dataset.py` confronta l'hash del *contenuto* dei pixel.
- **Strisce di bordo uniformi.** Due strisce nere coincidono senza essere vicine: la soglia
  `min_strip_std` le esclude.
- **Difetto nella zona di sovrapposizione.** Un difetto che cade nei 32 px compare in
  entrambe le patch adiacenti ed è la forma più grave di leakage; non va confuso con una
  coppia mancata.

## 6. Cosa è stato provato e cosa no

- Provato su dati sintetici con verità nota (split per patch e per rotolo): identificazione
  esatta della polvere, controlli di coerenza, coppie di bordo trovate (182 su 185; le 3
  mancanti sono un artefatto del generatore), quota cross-split 0.43 contro 0.41 atteso con
  split per patch e 0.0 con split per rotolo.
- **Non provato**: il backbone `resnet18` (torch non era disponibile), il dataset reale, il
  formato reale delle maschere (`mask_suffix` è un'ipotesi).
- Il backbone `simple` (34 numeri) non sostituisce il backbone dei modelli: per la tesi fissa
  in anticipo quello giusto.
- I risultati sui dati sintetici (AUROC ~1.0, ecc.) non dicono nulla sul tuo dataset.
