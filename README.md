# Guida all'Oracolo Self-Hosted (LLM_HOST)

Benvenuto nella cartella `LLM_HOST/`. Questa directory contiene il necessario per avviare un **Oracolo LLM locale** (Self-Hosted) da utilizzare all'interno del framework REFINE.

---

## 1. Cos'è cambiato? (Il nuovo paradigma)

In passato, far generare output JSON corretti a modelli locali (come Mistral o Qwen) era molto difficile: i modelli tendevano a inserire testo discorsivo (es. *"Ecco il tuo JSON:"*) o a sbagliare la sintassi, costringendo il framework a scartare moltissimi dati.

Per risolvere questo problema, **abbiamo eliminato tutto il codice Python personalizzato per il server**. 
Ora utilizziamo **vLLM**, un motore di inferenza professionale e open-source che offre due vantaggi enormi:

1. **API compatibili con OpenAI:** Il framework REFINE comunicherà con il tuo server locale esattamente come se stesse parlando con ChatGPT, senza bisogno di codice extra.
2. **Constrained Decoding (Modalità JSON rigorosa):** vLLM forza matematicamente il modello a restituire *esclusivamente* un JSON valido. Gli errori di parsing sono letteralmente azzerati.

---

## 2. Il Modello Consigliato: Qwen/Qwen3-14B

Il modello ideale per il task di validazione e generazione delle triple è **`Qwen/Qwen3-14B`**.

**Perché proprio questo modello?**
- **È un modello performante e versatile:** Qwen 3 14B è un modello di grandi dimensioni (14 miliardi di parametri) addestrato per seguire le istruzioni (Instruction-Tuned), il che lo rende molto efficace nel comprendere e aderire ai prompt strict, riducendo le allucinazioni e gli errori di formattazione.
- **Precisione chirurgica:** Essendo un modello Google addestrato per seguire le istruzioni (Instruction-Tuned), obbedirà ciecamente ai nostri prompt strict senza allucinare i nomi delle relazioni.
- **Requisiti Hardware:** Se caricato in quantizzazione a 4-bit (consigliato per ottimizzare la VRAM), richiede circa 8-10 GB di VRAM, girando comodamente su una singola GPU NVIDIA (es. RTX 3060 12GB, RTX 3090, 4090, o A10G/A100).

---

## 3. Guida all'Avvio (Step-by-Step)

Segui questi passaggi per avviare il server sul tuo computer/server Linux dotato di GPU NVIDIA.

Nota: per supportare modelli di ultima generazione come `google/gemma-4-26B-A4B-it` si raccomanda l'uso di Python 3.10 o superiore (l'ambiente attuale è basato su Python 3.13+).

### Step 0: Token HuggingFace per modelli gated
Il modello `google/gemma-4-26B-A4B-it` è gated, quindi devi avere un token HuggingFace valido con accesso già approvato al modello. Il modo consigliato è salvarlo in [LLM_HOST/.env](LLM_HOST/.env) e lasciare che [LLM_HOST/start_server.sh](LLM_HOST/start_server.sh) lo carichi automaticamente.
<!-- Qwen/Qwen3-14B non è un modello gated, quindi questa sezione può essere ignorata a meno che non si scelga un modello gated in futuro. -->
Variabili attese nel file `.env`:
```bash
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...
```

Se preferisci una sessione temporanea, puoi anche esportarle nel terminale prima di avviare lo script.

### Step A: Preparazione dell'ambiente
È caldamente consigliato usare un ambiente virtuale (Conda o venv).
```bash
# Crea un ambiente virtuale e attivalo
python3 -m venv vllm_env
source vllm_env/bin/activate

# Installa uv (raccomandato da vLLM) e le dipendenze
pip install uv
uv pip install -r requirements.txt
```
(Il file `requirements.txt` ora pinna `vllm==0.10.2`, perché la serie `0.11.x` richiede un runtime Python più recente e sul server disponibile stiamo usando Python 3.9.5.)*

### Step B: Avvio del Server
Abbiamo preparato uno script `start_server.sh` che lancia vLLM con le impostazioni di memoria ottimizzate.

```bash
# Rendi lo script eseguibile (solo la prima volta)
chmod +x start_server.sh

# Avvia il server
./start_server.sh
```

Il server scaricherà i pesi del modello (solo al primo avvio) e si metterà in ascolto sulla porta `8000`. Quando vedrai la scritta `Uvicorn running on http://0.0.0.0:8000`, l'Oracolo sarà pronto a ricevere richieste.

Per tenerlo attivo in background, usa `tmux`:
```bash
tmux new -s llm_host
./start_server.sh
```

Poi stacca la sessione con `Ctrl-b` seguito da `d`.

---

## 4. Testare il Server

Puoi verificare che il server funzioni e che la modalità JSON sia attiva aprendo un altro terminale e lanciando questo comando `curl`:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemma-4-26B-A4B-it",
       "messages": [ 
         {"role": "user", "content": "Restituisci un JSON con una chiave \"status\" e valore \"ok\"."}
       ],
       "temperature": 0.0
     }'
```
Se ti risponde con un JSON pulito, l'oracolo funziona perfettamente!

Per un controllo più comodo sulla validità JSON puoi usare anche `jq`:
```bash
curl -s -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemma-4-26B-A4B-it",
       "messages": [ 
         {"role": "user", "content": "Restituisci un JSON con una chiave status e valore ok."}
       ],
       "temperature": 0.0
     }' | jq .
```

---

## 5. Come lo usa il framework REFINE?

Per dire a REFINE di usare il tuo nuovo server locale, ti basterà specificare il nome del preset quando lanci la pipeline. 

Ad esempio, da riga di comando:
```bash
python -m examples.score_bridge.run_full_pipeline \
    your_dataset \
    --oracle-mode llm \
    --oracle qwen3_14b_local \
    --rounds 5
```

### 💡 Il trucco del suffisso "_local"
Hai notato che il preset si chiama `gemma4_26b_local`? Il suffisso `_local` è una parola magica per il framework REFINE.
Quando il framework vede un oracolo che finisce con `_local`, applica **automaticamente** queste regole di sicurezza nel file `ds_active/oracles_llm.py`:
1. Invia i sample da validare **uno alla volta** (Single-Sample Validation) invece che a blocchi (Batching), per evitare che il modello si confonda.
2. Usa un formato di JSON minimale e strict.
3. Indica a vLLM di usare la modalità constrained.

In questo modo, avrai la stabilità di GPT-4, ma con la privacy e il costo zero di un modello in locale!