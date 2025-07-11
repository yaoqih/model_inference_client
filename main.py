"""
Main entry point for the Model Inference Client API.
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .api.routes import router as api_router
from .services.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the FastAPI application.
    """
    # Code to run on startup
    print("Model Inference Client API starting up...")
    yield
    # Code to run on shutdown
    print("Model Inference Client API shutting down...")
    model_manager.shutdown()


app = FastAPI(
    title="Model Inference Client API",
    description="An API to start, stop, and monitor model inference servers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1", tags=["Model Management"])

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint for health checks.
    """
    return {"status": "ok", "message": "Welcome to the Model Inference Client API"}

if __name__ == "__main__":
    # This block allows running the app directly for development.
    # For production, it's recommended to use a process manager like Gunicorn.
    uvicorn.run(app, host="0.0.0.0", port=6004)
