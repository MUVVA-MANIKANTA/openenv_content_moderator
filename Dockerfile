FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (default for HF Spaces)
EXPOSE 7860

# Run the app using uvicorn
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
