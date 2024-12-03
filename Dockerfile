# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && apt-get clean

# Copy application files
COPY app.py /app/
COPY best1.pt /app/

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pillow ultralytics requests cloudinary

# Create temporary folder for image processing
RUN mkdir /app/temp

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
