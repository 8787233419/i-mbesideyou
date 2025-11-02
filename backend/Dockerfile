FROM nikolaik/python-nodejs:python3.11-nodejs20

WORKDIR /app

# Copy current backend folder contents
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node dependencies (if needed)
# RUN npm install

EXPOSE 8000

# Run both backend and whatsapp bot
CMD ["bash", "-c", "node whatsapp.js & uvicorn main:app --host 0.0.0.0 --port 8000"]
