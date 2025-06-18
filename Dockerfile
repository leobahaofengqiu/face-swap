FROM python:3.12-slim

# Install system dependencies for dlib and face_recognition, including a newer CMake
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && wget -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh \
    && chmod +x cmake-linux.sh \
    && ./cmake-linux.sh --skip-license --prefix=/usr/local \
    && rm cmake-linux.sh

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
