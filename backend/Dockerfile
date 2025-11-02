FROM nikolaik/python-nodejs:python3.11-nodejs20

# Set working directory
WORKDIR /app

# Copy only dependency files first (for caching)
COPY requirements.txt .
COPY package*.json ./

# Install Python and Node dependencies (cached if unchanged)
RUN pip install --no-cache-dir -r requirements.txt
RUN npm install --omit=dev

# Copy the rest of the app
COPY . .

# Expose the app port
EXPOSE 8000

# Start both backend and WhatsApp bot
CMD ["bash", "-c", "node whatsapp.js & uvicorn main:app --host 0.0.0.0 --port 8000"]
