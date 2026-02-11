FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "flask_app/app.py"]