# APAN5560-Assignment3


## How to Run


Option 1: Local (without Docker)

uv run python app/train_gan.py      # Train the GAN model

uv run uvicorn app.api:app --reload # Start API locally



Option 2: With Docker

docker compose build

docker compose up



Then open in browser:

Health: http://127.0.0.1:8000/health

Docs: http://127.0.0.1:8000/docs

PNG: http://127.0.0.1:8000/gan/generate.png?n=16&seed=42&nrow=4
