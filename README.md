GTM Website Analyzer — Phase 2: Crawler → Chunking → Embeddings → RAG Retrieval

Running locally 

bash
cd backend
python3 -m venv venv
source venv/bin/activate          
pip install -r requirements.txt
playwright install chromium       
cp .env.example .env

docker run -d --name gtm-postgres -p 5432:5432 \
  -e POSTGRES_USER=gtm_user -e POSTGRES_PASSWORD=gtm_password -e POSTGRES_DB=gtm_analyzer \
  pgvector/pgvector:pg16

uvicorn app.main:app --reload

Try it via `http://localhost:8000/docs`, or:

curl -X POST http://localhost:8000/api/websites \
  -H "Content-Type: application/json" \
  -d '{"url": "https://coirei.com", "max_pages": 100}'

curl http://localhost:8000/api/websites/<id>

curl -X POST http://localhost:8000/api/websites/<id>/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What pricing plans do they offer?", "top_k": 5}'


Running with Docker Compose (untested in this sandbox — see above)

bash
cp backend/.env.example backend/.env
docker compose up --build

bash
cd backend
pip install -r requirements.txt
pytest tests/ -v

