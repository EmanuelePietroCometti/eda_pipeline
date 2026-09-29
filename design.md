# Design dell'EDA del dataset

## 1. Scopo

Verificare che il dataset sia adatto a **addestrare e valutare** modelli di
anomaly detection sulle patch tessili, in particolare SuperSimpleNet, che usa i
difetti (con maschera) anche in fase di training. L'EDA deve dire se:

1. la valutazione sul test è affidabile (assenza di leakage tra train e test);
2. train e test descrivono la stessa popolazione (assenza di domain shift);
3. i difetti sono abbastanza grandi da essere localizzati alla risoluzione dei
   modelli;
4. il problema è separabile già prima di addestrare qualsiasi modello.

L'analisi è **descrittiva**: non serve a dimostrare ipotesi, ma a dichiarare i
limiti entro cui si interpretano le metriche dei modelli.

## 2. Dataset e definizioni

Layout (diverso da MVTec: il train contiene anche i difetti):

```
train/{good, <difetto>, ...}
test/{good, <difetto>, ...}
ground_truth/{good, <difetto>, ...}      # una maschera per immagine
```

I **nomi delle classi di difetto cambiano da tessuto a tessuto**. Ogni tessuto è
quindi un dataset a sé, analizzato separatamente; non si mettono in comune le
classi tra tessuti diversi. Una tabella di configurazione associa a ogni
tessuto i nomi delle sue cartelle di difetto.

Patch da 512×512 px, stride 480 px, quindi **32 px di sovrapposizione** tra patch
consecutive dello stesso rotolo; risoluzione 0.05 mm/px; i modelli ricevono
256×256 px.

### Polvere e versioni del dataset

Le immagini con polvere sono dentro `good` e **non hanno un nome che le
distingua**. Esistono tre versioni del dataset, che differiscono per dove si
trova la polvere:

| Versione | Polvere in `train/good` | Polvere in `test/good` |
|---|---|---|
| V0 | no | no |
| V1 | no | sì |
| V2 | sì | sì |

La polvere **ha sostituito** parte dei `good`, sia in validazione sia in
training, per mantenere costante la numerosità del dataset. Le versioni non sono
quindi sottoinsiemi l'una dell'altra: per ogni split, V1 = V0 − R1 + D1 e
V2 = V0 − R2 + D2, dove R sono i `good` rimossi e D le immagini con polvere
aggiunte.

**[DA CONFERMARE]** Ipotesi di lavoro, da verificare (§3): le uniche differenze tra
le versioni sono i `good` rimossi (R) e le immagini con polvere aggiunte (D), tutte
in `good`; difetti e maschere sono identici; il train di V1 è identico a quello
di V0.

Se l'ipotesi regge, la polvere si identifica senza annotazione manuale: sono le
immagini presenti in una versione con polvere e assenti in V0 **nello stesso
split**. Il confronto è **per contenuto** (hash dei pixel), non per nome, perché i
nomi potrebbero essere cambiati tra le versioni. Se l'ipotesi non regge, serve un
file `dust.csv` (colonna `filename`) compilato a mano.

**Tessuto.** Il tessuto nero è l'unico con polvere sufficiente per un'analisi EDA.
Le analisi che coinvolgono la polvere si fanno **solo sul nero**; per gli altri
tessuti si analizza V0, senza affermazioni sulla polvere. Le classi di difetto
cambiano da tessuto a tessuto, quindi i tessuti restano dataset separati.

**[DA CONFERMARE]** Numero di immagini con polvere nel nero, per split e per
versione (decide se un confronto è quantitativo o solo descrittivo, §5), e se la
polvere del test è la stessa in V1 e V2.

Conseguenze, fissate in anticipo:

- V0 è la **linea di base**: senza polvere, mostra lo shift tra train e test che
  esiste indipendentemente da essa.
- In V1 la polvere è solo nel test: confrontare `train/good` con `test/good`
  mescola shift reale e polvere. I gruppi sotto la separano sempre.
- In V2 la polvere è anche nel training: il modello la vede come normale. La
  misura da riportare è se la tollera senza dare falsi positivi.
- Poiché la polvere sostituisce dei `good` a numerosità costante, le versioni con
  polvere hanno **meno `good` puliti** a parità di totale. Una differenza tra
  versioni può quindi dipendere anche dal minor numero di `good` puliti, non solo
  dalla polvere: si dichiara e si riportano le numerosità di ogni gruppo.
- Le analisi che non dipendono dalla polvere (maschere e leakage tra difetti)
  si calcolano **una volta sola**, su V0, perché difetti e maschere sono identici
  nelle versioni (controllo di §3). Domain shift, baseline kNN, galleria e le
  coppie di leakage che coinvolgono `good` o polvere si calcolano per versione.

### Gruppi

Sei gruppi, usati in tutto il documento. `dust` viene dal confronto tra versioni
(o da `dust.csv`). In V0 i gruppi `*_dust` sono vuoti; in V1 è vuoto `train_dust`.

| Gruppo | Contenuto | Ruolo |
|---|---|---|
| `train_good` | `train/good` senza polvere | riferimento di normalità (pulito) |
| `train_dust` | `train/good` con polvere (solo V2) | normalità con polvere |
| `train_def` | `train/<difetto>` | supervisione (SuperSimpleNet) |
| `test_good` | `test/good` senza polvere | negativi della valutazione |
| `test_dust` | `test/good` con polvere | negativi con polvere |
| `test_def` | `test/<difetto>` | positivi della valutazione |

**[DA CONFERMARE]** nome e formato delle maschere (stesso stem dell'immagine?
suffisso? file vuoti per `good` o cartella assente?) e la classe di difetto che
somiglia di più alla polvere in ogni tessuto (serve alla galleria, §4.5).

## 3. Controlli di integrità (prima di ogni analisi)

Falliscono in modo esplicito, non si correggono in silenzio.

| Controllo | Esito atteso |
|---|---|
| Ogni immagine in `difetto*` ha una maschera | 100% |
| Le maschere di `good` sono vuote | 100% (una maschera non vuota è un'etichetta sbagliata) |
| Maschera e immagine hanno la stessa dimensione | 100% |
| Maschere di difetto non vuote | 100% |
| Nomi file unici tra i gruppi | nessun duplicato |
| Numerosità per split e classe uguale in V0, V1 e V2 | 100% |
| Le immagini di V0 assenti in V1 o V2 (stesso split) stanno tutte in `good` | 100% |
| Le immagini presenti in V1 o V2 e assenti in V0 (stesso split) stanno tutte in `good` | 100% |
| Per split, numero di immagini rimosse = numero di immagini aggiunte | 100% |
| `train/` di V1 identico a `train/` di V0 | 100% |
| Immagini e maschere dei difetti identiche tra le versioni (per contenuto) | 100% |
| Le immagini aggiunte non compaiono in V0 con altro nome | nessuna |
| Numero di immagini con polvere per versione e split | riportato in tesi |
| Relazione tra la polvere del test di V1 e di V2 (identica o no) | riportata in tesi |
| Tabella dei conteggi split × classe | riportata in tesi, incluse le proporzioni good/difetto nel train |

## 4. Analisi

Per ognuna: domanda, metodo, criterio di lettura fissato in anticipo, decisione
che ne segue.

### 4.1 Statistiche delle maschere

- **Domanda.** Quanto sono grandi i difetti, e train e test li contengono di
  dimensione simile?
- **Metodo.** Componenti connesse (8-connettività) di ogni maschera. Per ogni
  componente: area in px, area in % della patch, bounding box, extent. Le aree
  sono riscalate a 256 px per confrontarle con le celle dei backbone (stride 8 →
  64 px², stride 16 → 256 px²).
- **Output.** Per gruppo (`train_def`, `test_def`) e classe: numero di immagini,
  numero di componenti per immagine, mediana e range interquartile dell'area,
  frazione di componenti sotto una cella. ECDF dell'area (asse log) con le soglie
  di cella; box plot dell'area in %.
- **Lettura.** Una frazione alta di componenti sotto una cella rende attesi un
  AP-loc basso e la necessità di metriche a livello di pixel. Il confronto
  train/test è descrittivo (ECDF sovrapposte, rapporto tra le mediane); niente
  p-value. Con meno di 20 componenti per classe si mostrano solo i punti.
- **Decisione.** Se le distribuzioni di train e test divergono in modo evidente,
  lo si dichiara come limite della valutazione.

### 4.2 Domain shift

- **Domanda.** Le immagini nominali (e i difetti) del test provengono dalla stessa
  popolazione del train?
- **Metodo.** Per ogni versione, tre confronti: `train_good` contro `test_good` (entrambi senza
  polvere, quindi il vero shift tra split), `train_good` contro `test_dust`
  (l'effetto della polvere da sola) e `train_def` contro `test_def`. Il primo
  confronto in V0 è la linea di base. Per ogni immagine: media e deviazione standard RGB e della
  luminosità, istogramma di luminosità a 64 bin. Cohen's d con intervallo di
  confidenza bootstrap. Un classificatore (regressione logistica su feature
  standardizzate, PCA a 20 componenti) distingue train da test con AUC in
  validazione incrociata **raggruppata** per gruppo di patch adiacenti, confrontato
  con la distribuzione nulla ottenuta permutando le etichette.
- **Lettura.** |d| ≥ 0.8 è uno spostamento grande, ≈ 0.5 medio. AUC entro il 95°
  percentile del nullo: nessuno shift rilevabile; AUC oltre: le due popolazioni si
  distinguono. Il classificatore non si esegue con meno di 20 immagini per classe: con la sola
  polvere di un tessuto, di norma poche, si riportano solo d e istogrammi.
- **Decisione.** Uno shift già sui campioni buoni spiega falsi positivi che non
  dipendono dal modello e va riportato prima dei risultati dei modelli.

### 4.3 Leakage e duplicati

- **Domanda.** Esistono patch di train e test che condividono pixel?
- **Metodo.** Confronto **esatto** dei bordi: l'ultima striscia da 32 px di una
  patch contro la prima della successiva, in orizzontale e in verticale, tramite
  hash dei valori di grigio. In più, hash percettivo (pHash, Hamming ≤ 4) con
  conferma per correlazione. Le coppie formano gruppi (componenti connesse).
- **Output.** Numero di coppie per combinazione di gruppi:
  (`train_good`,`test_good`), (`train_def`,`test_def`), (`train_def`,`test_good`)
  e così via. Quota di coppie a cavallo di train e test, confrontata con quella
  attesa se lo split fosse per patch casuale.
- **Lettura.** Quota osservata vicina all'attesa: split per patch (leakage).
  Vicina a 0: split per rotolo o frame. Le coppie che coinvolgono `good` o polvere si ricalcolano per ogni versione,
  perché la composizione cambia. Le coppie tra difetti sono le più gravi,
  perché un difetto sul bordo compare con la sua maschera in entrambe le patch.
- **Decisione.** Se esiste anche una sola coppia difetto–difetto tra train e test,
  le metriche di SuperSimpleNet sul test vanno riportate con questo avviso, e va
  valutato uno split per rotolo. Se nel dataset sono state tenute solo alcune patch,
  i vicini mancanti non si trovano e il leakage è sottostimato: si dichiara.

### 4.4 Baseline di distanza kNN

- **Domanda.** Quanto è separabile il problema senza addestrare nulla?
- **Metodo.** Punteggio di anomalia = distanza media dai k vicini più prossimi nel
  riferimento (feature standardizzate sul riferimento). I difetti del train non
  entrano mai nel riferimento. Il riferimento è `train/good` **della versione
  analizzata**: pulito in V0 e V1 (stesse immagini), con polvere in V2. Le immagini
  del riferimento sono punteggiate in leave-one-out. `k = 1` principale, `k = 5`
  secondario.
- **Contrasti fissati in anticipo.** V1 contro V0: effetto della polvere nel test con
  riferimento pulito (quanta polvere è vista come anomala). V2 contro V1: effetto
  della polvere nel riferimento (se vederla in training la fa tollerare).
  I difetti da valutare sono gli stessi nelle tre versioni (§3); i `good` puliti
  no (sostituiti dalla polvere a numerosità costante), e i contrasti vanno letti
  tenendone conto.
- **Backbone.** **[DA CONFERMARE]** un solo backbone, scelto in anticipo e uguale a
  quello dei modelli (stesso pooling globale di layer2 e layer3).
- **Metriche.** AUROC e AP con intervallo di confidenza bootstrap stratificato
  (1000 ricampionamenti, seme 42), per: tutti i difetti contro `test_good`, ogni
  classe contro `test_good`, tutti i difetti contro `test_good` + `test_dust`
  (l'insieme di negativi che il modello incontra davvero), `test_dust` contro
  `test_good` (quanto la polvere sembra anomala). Si riportano le numerosità
  accanto a ogni cifra.
- **Lettura.** È un **limite inferiore**: un vettore per immagine con pooling
  globale diluisce i difetti piccoli. Non è PatchCore e non va presentato come tale.
- **Robustezza.** Se il leakage è presente, le metriche vanno ricalcolate senza i
  test con vicini nel train e senza i train dei gruppi condivisi.

### 4.5 Galleria qualitativa

- **Domanda.** Come appaiono i difetti, e come si distinguono da campioni nominali
  simili?
- **Metodo.** Confronto tra la classe di difetto più simile alla polvere (scelta
  per tessuto, §2) e `test_dust`, con la polvere identificata in §2. Campione casuale a seme
  fisso (42), 12 per classe, dal test (tutte le immagini se sono meno di 12; con
  poca polvere si mostrano tutte).
  Patch intere con contorno della maschera e ritaglio di 160 px attorno alla
  componente più grande. La polvere non ha maschera (le maschere di `good` sono
  vuote): l'area della polvere non è misurabile e non si riporta alcun confronto
  di area tra le due classi.
- **Lettura.** Nessun test statistico. Il campione è casuale e non scelto a mano,
  e l'elenco dei file va nell'appendice.

## 5. Parametri fissati

Da riportare in tesi; da modificare solo dichiarandolo.

| Parametro | Valore |
|---|---|
| Seme | 42 |
| Patch / stride / sovrapposizione | 512 / 480 / 32 px |
| Risoluzione dei modelli | 256×256 |
| Soglie di cella | 8 e 16 px (stride di layer2 e layer3) |
| Connettività maschere | 8 |
| Istogramma di luminosità | 64 bin |
| Bootstrap (d, AUROC) | 1000 ricampionamenti, IC 95% |
| Classificatore di dominio | PCA 20, regressione logistica bilanciata, CV a 5 fold raggruppata, 50 permutazioni |
| Numerosità minima per confronti quantitativi | 20 |
| pHash | Hamming ≤ 4, correlazione ≥ 0.98 |
| kNN | k = 1 (principale), 5; riferimento solo `train_good` |
| Galleria | 12 per classe, ritaglio 160 px |
| Identificazione della polvere | differenza per contenuto tra versioni (fallback `dust.csv`) |
| Versioni | V0, V1, V2; riferimento kNN = `train/good` della versione |

## 6. Limiti dichiarati in anticipo

- L'EDA non dimostra che i modelli generalizzino: dice solo se i dati permettono di
  fidarsi delle metriche.
- Con poche immagini per classe gli intervalli sono larghi: si guardano loro, non le
  stime puntuali.
- Nessuna analisi sostituisce una valutazione con split per rotolo se il dataset
  attuale non lo è.
- La polvere è identificata solo dalla differenza tra versioni (o da `dust.csv`):
  se le versioni non differiscono solo per la polvere, l'identificazione è
  sbagliata e si propaga a tutte le analisi che la separano dal resto di `good`.
- La polvere è studiata solo sul tessuto nero: nessuna affermazione sugli altri
  tessuti riguardo alla polvere.
- I tessuti sono analizzati separatamente: non si dichiarano risultati comuni
  senza averli osservati in ciascuno.
- Le feature del kNN sono globali: un risultato basso sui micro-difetti è atteso e
  non è un giudizio sul dataset.

## 7. Ordine di implementazione

0. Controllo di coerenza tra le versioni e identificazione della polvere (§2, §3).
1. Indice del dataset e controlli di integrità (§3).
2. Statistiche delle maschere (§4.1).
3. Leakage esatto (§4.3): dà l'informazione più importante per leggere tutto il resto.
4. Domain shift (§4.2).
5. Baseline kNN (§4.4), dopo aver fissato il backbone.
6. Galleria (§4.5).

Ogni passo si chiude con un controllo su un caso noto (ricalcolo a mano di alcune
aree, coppie di bordi verificate a occhio) prima di passare al successivo.

## 8. Cosa riportare in tesi

- Tabella dei conteggi split × classe e proporzioni nel train.
- Esito dei controlli di integrità.
- Per ogni analisi: parametri, cifre con intervalli, numerosità, e se il criterio
  fissato in §4 è stato soddisfatto.
- Deviazioni dal progetto, se ci sono, con motivazione.
