# Code

Replica di Boucher, Rendall, Ushchev e Zenou (2024), versione consolidata v5.
Genera dati sintetici di "scuole" a partire dal modello strutturale con parametri veri noti, stima il modello tramite GMM e produce le figure diagnostiche e quelle in stile diapositiva.

## Come eseguire tutto

```bash
python run_all_v5.py
```

Questo esegue in ordine `teaching_steps_v5.py`, poi la stima CUE completa, poi `Figures_v5.py` e `SlideFigures_v5.py`.
Ogni script puo anche essere eseguito separatamente, ma tutti assumono che `DGP_v5.py` abbia gia generato i dati in `data/generated_v5/`.

## Script

**`DGP_v5.py`** e il processo generatore dei dati. Costruisce la rete di amicizie per ogni scuola, le covariate degli studenti e il GPA di equilibrio a partire dal modello strutturale (lambda, beta, delta noti). Salva tutto in `data/generated_v5/`. Espone `build_environment`, `save_environment`, `load_environment`.

**`Estimation_v5.py`** e il modulo di stima (GMM concentrato in due fasi piu CUE). Nella prima fase predice il GPA usando le covariate proprie e gli effetti fissi di scuola, poi costruisce gli strumenti (`Shat`, `Dhat` e varianti selezionabili) a partire dai valori predetti, infine cerca lambda, beta e delta con errori standard robusti per cluster di scuola. Include test (J di sovraidentificazione, Anderson Rubin, restrizione LIM) e analisi di policy (giocatore chiave, frontiera di identificazione, Monte Carlo).

**`Figures_v5.py`** genera le figure diagnostiche `fig1` fino a `fig11` in `figures_v5/` a partire dai dati generati e dai risultati di stima salvati.

**`SlideFigures_v5.py`** genera le figure in stile diapositiva didattica `slide1` fino a `slide6`, ovvero grafi di rete, matrice di adiacenza, triade intransitiva, scatter di coppie raggruppate, distribuzione del risultato e moltiplicatore sociale.

**`teaching_steps_v5.py`** e uno script didattico passo per passo che mostra i passaggi interni dello stimatore in due fasi, riutilizzando le funzioni di `DGP_v5` e `Estimation_v5`.

**`run_all_v5.py`** e l'orchestratore che esegue tutta la pipeline con un solo comando.

## Cartelle di dati e output (generate automaticamente)

**`data/generated_v5/`** contiene i dati sintetici, cioe `students.csv`, `true_parameters.json`, `metadata.json` e le reti per scuola (`G_school_XX.npz` normalizzata per riga, `raw_G_school_XX.npz` non normalizzata).

**`outputs_v5/`** contiene i risultati della stima, cioe `estimation_results.json`, `estimation_comparison.csv` (stimato contro vero) e `final_estimates_with_cluster_se.csv`.

**`figures_v5/`** contiene le figure PNG prodotte da `Figures_v5.py` e `SlideFigures_v5.py`.

**`__pycache__/`** contiene il bytecode compilato di Python e si puo eliminare senza problemi.

## Ordine delle dipendenze

```
DGP_v5.py  >  Estimation_v5.py  >  Figures_v5.py / SlideFigures_v5.py
                  ^
          teaching_steps_v5.py (usa entrambi)
```
