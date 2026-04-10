FROM python:3.11-slim

WORKDIR /app

# Copy requirement and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port (Render sets PORT environment variable)
EXPOSE 8000

# By default, run the web service (this can be overridden by Render commands)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
