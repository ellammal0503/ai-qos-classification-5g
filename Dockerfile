# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy local code to container
COPY . .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
