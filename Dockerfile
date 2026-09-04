FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port (7860 for Hugging Face Spaces, 8000 default)
EXPOSE 7860
EXPOSE 8000

# Start command
CMD ["python", "api.py"]
