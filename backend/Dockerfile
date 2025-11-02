# Use a base image with both Python and Node
FROM nikolaik/python-nodejs:python3.11-nodejs20

WORKDIR /app

# Copy backend files
COPY ./backend /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node dependencies (if you use any in whatsapp.js)
RUN npm install

# Expose FastAPI default port
EXPOSE 8000

# Run both backend and whatsapp script
CMD ["bash", "-c", "node whatsapp.js & uvicorn main:app --host 0.0.0.0 --port 8000"]
