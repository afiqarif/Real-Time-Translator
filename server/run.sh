#!/bin/bash
echo "================================================="
echo "   STARTING COSYVOICE SERVER + PUBLIC TUNNEL"
echo "================================================="

# 1. Kill old processes to free up memory
echo ">>> 1. Cleaning up old processes..."
pkill -f server.py
pkill -f "lt --port"

# 2. Start the Server in the background
echo ">>> 2. Starting Python Server..."
python server.py > server.log 2>&1 &

# 3. Wait for the model to load
echo ">>> 3. Waiting 20 seconds for AI to load..."
sleep 20

# 4. Start the Tunnel and show URL
echo "================================================="
echo ">>> DONE! COPY THE URL BELOW FOR YOUR LAPTOP:"
echo "================================================="
# Try global lt, fall back to npx if missing
if command -v lt &> /dev/null; then
    lt --port 8080
else
    npx localtunnel --port 8080
fi
