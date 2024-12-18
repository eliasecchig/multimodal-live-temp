# Run Frontend

```bash
npm install
npm start
```

# Run Backend
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.server:app --host 0.0.0.0 --port 8000  --reload 
```






















