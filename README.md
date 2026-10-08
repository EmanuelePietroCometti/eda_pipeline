# EDA del dataset tessile

Analisi esplorativa delle patch tessili per l'anomaly detection. Tre analisi, tutte descrittive:

1. **Numerosità per gruppo e per versione** del dataset, con l'**area mediana** dei difetti.
2. **Buoni e polvere**: `train_good`, `test_good`, `train_dust`, `test_dust` nella stessa figura,
   con le proiezioni PCA, t-SNE e UMAP affiancate (la polvere si sovrappone ai buoni?).
3. **Buoni vs difetti** senza polvere, con le stesse tre proiezioni in 2D e in 3D (sono separabili?).

## Dati

```
train/{good, <difetto>, ...}
test/{good, <difetto>, ...}
ground_truth/{good, <difetto>, ...}      # una maschera per immagine, stesso nome (più mask_suffix)
```

Patch 512×512 px, i modelli ricevono 256×256. I nomi delle classi di difetto cambiano da tessuto
a tessuto: ogni tessuto è un dataset a sé, con i propri percorsi nel `config.yaml`.

### Versioni e polvere

La polvere sta dentro `good` e non ha un nome che la distingua. Tre versioni, che differiscono
per dove si trova:

| Versione | Polvere in `train/good` | Polvere in `test/good` |
|---|---|---|
| V0 | no | no |
| V1 | no | sì |
| V2 | sì | sì |

La polvere ha **sostituito** parte dei `good`, quindi le numerosità per split restano uguali ma le
versioni con polvere hanno meno `good` puliti. La polvere si identifica per differenza: sono le
immagini di `good` presenti in una versione e assenti in V0 nello stesso split, confrontate **per
contenuto** (hash dei pixel), non per nome. Il confronto è affidabile solo se le versioni differiscono
soltanto per i `good` rimossi e la polvere aggiunta (vedi Limiti).

Sei gruppi: `train_good`, `train_dust`, `train_def`, `test_good`, `test_dust`, `test_def`
(`good` = senza polvere). In V0 i gruppi `*_dust` sono vuoti; in V1 è vuoto `train_dust`.
La polvere si studia solo sul tessuto nero.

## Uso

```bash
pip install -r requirements.txt            # include umap-learn; per resnet18 servono anche torch e torchvision
python run.py                              # tutto, con config.yaml
python run.py --steps counts               # solo alcuni passi: counts projections
python run.py --config altro.yaml
```

Prima di lanciare: metti i percorsi delle versioni in `config.yaml` (V1 e V2 sono commentati: senza
di loro non c'è polvere e la figura buoni/polvere viene saltata), controlla `mask_suffix` e fissa
`embedding.backbone`. Prova senza dati reali: `python tools/make_synthetic.py --out /tmp/synth --rolls 10`,
poi punta il config a `/tmp/synth/V0`, `V1`, `V2`, con `mask_suffix: "_mask"` e `backbone: simple`.

## Cosa produce

Tutto sotto `general.out_dir` (default `results/`). Le tabelle sono CSV.

| File | Contenuto |
|---|---|
| `counts.csv`, `counts_wide.csv` | numerosità per versione, gruppo e classe, con l'area mediana dei difetti |
| `<versione>/figures/proj_polvere_vs_buoni.png` | PCA, t-SNE, UMAP di buoni e polvere, train e test (versioni con polvere) |
| `<versione>/figures/proj_buoni_vs_difetti.png`, `proj_buoni_vs_difetti_3d.png` | PCA, t-SNE, UMAP di buoni e difetti, senza polvere, in 2D e in 3D |
| `<versione>/projections/*.csv` | coordinate proiettate, una riga per immagine (`*_3d.csv` per il 3D) |
| `<versione>/index.csv` | indice delle immagini con il gruppo assegnato |

### Numerosità e area (`counts.csv`)

Una riga per versione, gruppo e classe. `class` è `-` per buoni e polvere, `ALL` per tutti i difetti
del gruppo, altrimenti la classe di difetto. `n_images` è il numero di immagini. Per i difetti l'unità
dell'area è la componente connessa della maschera (8-connettività), un "difetto": `n_defects`,
`area_px_median` (mediana in px della patch 512) e `area_pct_median` (mediana in % della patch).
Buoni e polvere non hanno maschera: area vuota. `counts_wide.csv` ha le versioni in colonna.

## Decisioni di progetto

* **Le proiezioni sono grafici, non misure.** t-SNE e UMAP non conservano le distanze globali e la
  PCA a 2-3 componenti mostra una parte della varianza (riportata sugli assi). Due gruppi che si
  sovrappongono in 2D possono essere separati nello spazio completo, e viceversa.
* **In 3D una sola prospettiva può ingannare**: la figura 3D è vista da un angolo fisso e va letta
  insieme a quella 2D.
* **Ogni figura si adatta ai soli gruppi che mostra.** Scaler fissato sui `train_good`; PCA, t-SNE e
  UMAP si adattano all'insieme mostrato, perché t-SNE non può proiettare punti nuovi.
* **Gli embedding si calcolano una volta per contenuto**: sono in cache
  (`embeddings_<backbone>_<size>.npz`); cancellala se cambi backbone o pesi.

## Limiti

* Embedding globali: un vettore per immagine diluisce i difetti piccoli, che nelle proiezioni
  possono cadere tra i buoni anche se sono distinguibili a livello di patch.
* La polvere è identificata per differenza da V0: è corretta solo se le versioni differiscono soltanto
  per i `good` rimossi e la polvere aggiunta (le numerosità in `counts.csv` aiutano a verificarlo:
  per split, i `good` mancanti rispetto a V0 devono essere tanti quanti la polvere).
* `backbone: simple` (34 numeri) serve solo a far girare l'EDA; per la tesi fissa lo stesso backbone dei modelli.
