#!/usr/bin/env bash
# Run MVP system chat with CrewMember and system monitoring
#
# This script sets up and runs the PersonaPlex system monitoring chat example.
# It demonstrates how CrewMember can discuss Thor system stats in natural conversation.
#
# Prerequisites:
#   - micromamba environment "chatty-buoy" activated
#   - PersonaPlex service running on localhost:8000
#   - FunctionGemma service running on localhost:8001
#   - Network access to both services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "════════════════════════════════════════════════════════════"
echo "  PersonaPlex System Monitoring Chat MVP"
echo "════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Check prerequisites
check_service() {
    local service_name=$1
    local port=$2
    local url="http://localhost:$port/v1/models"
    
    echo -n "Checking $service_name on port $port... "
    
    if timeout 2 curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

echo ""
echo "Prerequisites Check:"
echo "────────────────────────────────────────────────────────────"

# Check environment
if ! command -v micromamba &> /dev/null; then
    echo -e "${RED}✗ micromamba not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} micromamba available"

# Check environment exists
if ! micromamba env list | grep -q "chatty-buoy"; then
    echo -e "${RED}✗ chatty-buoy environment not found${NC}"
    echo "Create it with: micromamba create -f environment.yml"
    exit 1
fi
echo -e "${GREEN}✓${NC} chatty-buoy environment available"

# Check services
if ! check_service "PersonaPlex" 8000; then
    echo ""
    echo -e "${YELLOW}PersonaPlex not ready yet. Starting...${NC}"
    cd "$PROJECT_ROOT"
    docker compose -f docker/vllm/docker-compose.personaplex.yml up -d
    echo "Waiting for PersonaPlex to be ready (this may take 2-5 minutes)..."
    
    # Wait for PersonaPlex
    for i in {1..60}; do
        if check_service "PersonaPlex" 8000 2>/dev/null; then
            break
        fi
        echo -n "."
        sleep 5
    done
    echo ""
fi

if ! check_service "FunctionGemma" 8001; then
    echo ""
    echo -e "${YELLOW}FunctionGemma not ready yet. Starting...${NC}"
    cd "$PROJECT_ROOT"
    docker compose -f docker/vllm/docker-compose.gemma-function.yml up -d
    echo "Waiting for FunctionGemma to be ready (this may take 2-5 minutes)..."
    
    # Wait for FunctionGemma
    for i in {1..60}; do
        if check_service "FunctionGemma" 8001 2>/dev/null; then
            break
        fi
        echo -n "."
        sleep 5
    done
    echo ""
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All services ready! Starting chat..."
echo "════════════════════════════════════════════════════════════"
echo ""

# Run the chat example
cd "$PROJECT_ROOT"
micromamba run -n chatty-buoy python3 examples/personaplex_system_chat.py
