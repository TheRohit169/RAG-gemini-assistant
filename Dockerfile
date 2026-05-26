# Use a slim Python image to keep the build fast
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for FAISS and C-based libs
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Render expects
EXPOSE 10000

# Run the app
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "10000"]