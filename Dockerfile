# Use Python 3.12.1 as base image
FROM python:3.12.1

# Install system dependencies required for PyAudio and other packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    libasound2-dev \
    pulseaudio \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""

# Command to run the application
CMD ["python", "va.py"]