# Medical Chatbot Deployment

## Local Run
1. Add `medical_data.pdf` to the directory.
2. Run `pip install -r requirements.txt`.
3. Start backend: `python backend.py`.
4. Start frontend: `streamlit run frontend.py`.
5. Access at http://localhost:8501.

## Deploy on Render (Free)
1. Create GitHub repo with all files (exclude .env; set env vars in Render dashboard).
2. For backend: New Web Service > Python > Build: `pip install -r requirements.txt` > Start: `uvicorn backend:app --host 0.0.0.0 --port $PORT`.
3. For frontend: New Web Service > Python > Build: `pip install -r requirements.txt` > Start: `streamlit run frontend.py --server.port $PORT --server.headless true`.
4. Update BACKEND_URL in frontend.py to the Render URL.
5. Add GOOGLE_API_KEY in Render's Environment Variables.

Data persists in Render's disk; re-upload PDF if needed.
