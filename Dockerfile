FROM python:3.11-slim

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt pyproject.toml setup.py ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
