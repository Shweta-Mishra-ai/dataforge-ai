# 1. Base Image: Use a slim Python 3.11 Image
FROM python:3.11-slim

# 2. System dependencies (for ReportLab/FastAPI/Streamlit internals)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Setting the working directory
WORKDIR /app

# 4. Copying the Python requirements
COPY requirements.txt .

# 5. Installing the Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application
COPY . .

# 7. Expose Streamlit Port
EXPOSE 8501

# 8. Healthcheck to guarantee the container is running cleanly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Entrypoint command to start Streamlit with High-Performance options
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.headless=true"]
