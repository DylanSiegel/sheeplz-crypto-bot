#!/bin/bash

# scripts/deploy.sh

# Stop existing containers
docker-compose down

# Build Docker images
docker-compose build

# Pull latest dependencies (if using git)
git pull origin main

# Start containers
docker-compose up -d

# View logs
docker-compose logs -f
