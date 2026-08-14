FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir eventlet gunicorn

# Copy app
COPY . .

EXPOSE 5000

# Init DB and start server
CMD ["sh", "-c", "python init_db.py && gunicorn wsgi:app --worker-class eventlet -w 1 --bind 0.0.0.0:5000"]
