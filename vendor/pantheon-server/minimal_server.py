"""
Minimal Pantheon Server API for testing

This version bypasses the hanging pantheon-legends import to test the server infrastructure.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Pantheon Server (Minimal)",
    description="Cryptocurrency analysis server - minimal version for testing",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning basic server information"""
    return {
        "service": "Pantheon Server (Minimal)",
        "version": "0.1.0",
        "description": "Cryptocurrency analysis using Pantheon Legends",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "running",
        "note": "Minimal version for testing - pantheon-legends integration pending"
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "pantheon-server-minimal",
        "pantheon_legends": "pending",
        "note": "Server infrastructure is working"
    }

@app.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """Test endpoint to verify the server is responding"""
    return {
        "message": "Server is working!",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "root": "/",
            "health": "/health",
            "docs": "/docs",
            "test": "/test"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("🏛️  Starting Pantheon Server (Minimal)...")
    print(f"🌐 Server starting on http://{host}:{port}")
    print(f"📖 API docs available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, reload=True)
