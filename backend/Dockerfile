# Stage 1: Base image with both Python and Node
FROM python:3.11-slim

# Install Node.js
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Set workdir
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node dependencies (if whatsapp.js uses package.json)
RUN if [ -f package.json ]; then npm install; fi

# Expose Flask default port
EXPOSE 5000

# Run both Flask and Node together
CMD sh -c "python main.py & node whatsapp.js"
