PY ?= python

.PHONY: install train app all docker-build docker-up docker-down

install:
	$(PY) -m pip install -r requirements.txt

# Cleans data, runs EDA, trains + tunes 4 models, evaluates, saves artifacts to models/ and outputs/
train:
	$(PY) -m src.main

app:
	streamlit run app/streamlit_app.py

# Train, then launch the app
all: train app

# --- Docker: whole application in one container ---
docker-build:
	docker compose build

docker-up:
	docker compose up -d --build
	@echo "App running at http://localhost:8501"

docker-down:
	docker compose down
