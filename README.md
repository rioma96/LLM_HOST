# LLM_HOST - Mistral 7B Relation Oracle

Servizio FastAPI per usare un LLM come oracolo di relazione:
`dato (frase, entity_1, entity_2, relation_name)`, restituisce se la relazione è presente o no.

## 1) Setup venv locale

```bash
cd /home/utente/Documenti/barH/progetto/nuovo_disco/LLM_HOST
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Installazione dipendenze

Installa prima PyTorch CUDA compatibile con il tuo sistema/GPU, poi il resto:

```bash
cd /home/utente/Documenti/barH/progetto/nuovo_disco/LLM_HOST
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 3) Configurazione

```bash
cd /home/utente/Documenti/barH/progetto/nuovo_disco/LLM_HOST
cp .env.example .env
```

Variabili principali in `.env`:
- `MODEL_ID` default: `mistralai/Mistral-7B-Instruct-v0.2`
- `LOAD_IN_4BIT=true` per ridurre VRAM
- `MAX_NEW_TOKENS=96`

## 4) Avvio servizio

```bash
cd /home/utente/Documenti/barH/progetto/nuovo_disco/LLM_HOST
source .venv/bin/activate
python main.py
```

Il server parte su `0.0.0.0:8000` (configurabile da `.env`) ed è quindi raggiungibile esternamente via IP VM.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 5) Esempio richiesta

```bash
curl -X POST http://127.0.0.1:8000/v1/relation/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "Barack Obama was born in Honolulu.",
    "entity_1": "Barack Obama",
    "entity_2": "Honolulu",
    "relation_name": "place_of_birth",
    "relation_description": "entity_1 was born in entity_2"
  }'
```

Per il tuo framework active learning (prompt già costruito), usa endpoint raw:

```bash
curl -X POST http://155.185.5.37:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Given sentence and entities, return JSON {relation_present, confidence, reason_short}. Sentence: Barack Obama was born in Honolulu..."
  }'
```

Stato caricamento modello in RAM:

```bash
curl http://155.185.5.37:8000/v1/model/status
```

Compatibilità OpenAI-style (per `LlmClient`):

```bash
curl -X POST http://155.185.5.37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [
      {"role": "system", "content": "You are a relation verifier."},
      {"role": "user", "content": "Return valid JSON only. Items: ..."}
    ],
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

## 6) Smoke test rapido

Con server in esecuzione:

```bash
cd /home/utente/Documenti/barH/progetto/nuovo_disco/LLM_HOST
source .venv/bin/activate
python tests/smoke_request.py
```

## 7) Note rete VM (servizio visibile all'esterno)

Assicurati che la porta sia aperta nel firewall/security group della VM.

Esempio con UFW:

```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

Test remoto (da altra macchina):

```bash
python -c "import requests;print(requests.get('http://155.185.5.37:8000/health', timeout=10).text)"
```

## 8) Stato modello e spazio disco

- Il modello **non è pre-caricato** all'avvio server: viene scaricato/caricato al primo `POST` di inferenza.
- Per `mistralai/Mistral-7B-Instruct-v0.2` tieni almeno ~20GB liberi su `/` per download + cache + overhead.
- Se ricevi errore `No space left on device`, libera spazio o usa un modello più piccolo.

## 9) Config framework esterno (`LlmClient`)

Nel framework active learning:
- `base_url`: `http://155.185.5.37:8000/v1`
- `model`: `mistralai/Mistral-7B-Instruct-v0.2`
- `api_key`: opzionale; se vuoi enforcement imposta `OPENAI_API_KEY` nel `.env` del server.

Il tuo `oracle.py` e i prompt batch restano invariati: il servizio ora accetta `chat.completions` OpenAI-compatible.
