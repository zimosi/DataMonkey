#!/bin/bash

# Data Monkey Quick Start Script

echo "🐵 Data Monkey MVP - Quick Start"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}Error: backend/.env file not found${NC}"
    echo "Please create backend/.env with your OPENAI_API_KEY"
    echo "Example:"
    echo "OPENAI_API_KEY=your_key_here"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}Warning: Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Check ports
echo "Checking ports..."
check_port 8000
check_port 3000

echo ""
echo "Starting Data Monkey MVP..."
echo ""

# Start backend in background
echo -e "${GREEN}Starting Backend Server...${NC}"
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Start frontend in background
echo -e "${GREEN}Starting Frontend Server...${NC}"
cd auto_ml
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}✅ Data Monkey is starting!${NC}"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C and clean up
trap cleanup INT
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Done!"
    exit 0
}

# Wait for user to press Ctrl+C
wait
