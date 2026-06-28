# ✈️ Airlines Delay Prediction

A full-stack machine learning web application that predicts flight delays using LSTM and Transformer deep learning models.

## 🚀 Features

- Predicts flight delay based on airline, airport, and flight details
- Trained using LSTM and Transformer models
- REST API backend built with Python
- Modern frontend built with React + Vite

## 🛠️ Tech Stack

**Frontend**
- React
- Vite

**Backend**
- Python
- FastAPI / Flask
- TensorFlow / Keras (LSTM & Transformer models)
- Scikit-learn

## 📁 Project Structure
flight-delay-app/

├── backend/

│   ├── models/          # Trained ML models

│   ├── main.py          # API entry point

│   ├── predictor.py     # Prediction logic

│   ├── model_loader.py  # Model loading utilities

│   └── retrain.py       # Model retraining script

├── frontend/

│   ├── src/             # React source files

│   ├── public/          # Static assets

│   └── index.html

└── README.md

## ⚙️ Setup & Installation

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Models Used

- **LSTM** — Long Short-Term Memory network for sequential pattern learning
- **Transformer** — Attention-based model for improved accuracy

## 📊 Dataset

Flight delay data including airline codes, airport codes, and historical delay records.
> Dataset not included in repository due to size constraints.

## 🙌 Author

**Kavin** — [GitHub](https://github.com/Kavin-arch)
