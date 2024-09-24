FROM python:3.8-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app code
COPY . /app

# Set the working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/temp
RUN mkdir -p /app/results

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
