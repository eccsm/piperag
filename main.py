import asyncio
import logging
import os
import time
import gc
import resource
from typing import Union, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, BackgroundTasks, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader

# Enable uvloop for better async performance on Linux
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("UVLoop enabled for improved performance")
except ImportError:
    print("UVLoop not available, using standard event loop")


# Set resource limits to prevent memory issues
# These are soft limits that can be exceeded if necessary
def limit_memory():
    # Set soft limit to 6GB (leave 2GB for system)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 * 1024 * 1024, hard))
    print(f"Memory limit set to 6GB")


# Apply memory limits
try:
    limit_memory()
except Exception as e:
    print(f"Warning: Could not set memory limits: {e}")

# Set MLC optimization environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MLC_NUM_THREADS"] = "4"  # Optimized for your 16-core CPU with limited RAM
os.environ["TVM_NUM_THREADS"] = "4"  # Same as above
os.environ["MLC_USE_CUDA"] = "0"  # No GPU available

# Enable periodic garbage collection to prevent memory leaks
gc.enable()

from services.image_recognition_service import ImageRecognitionService
from services.service_factory import ServiceFactory
from services.chat_service import ChatService
from config import Config

# Configure logging to be less verbose
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("piperag")

# Reduce logging verbosity for common libraries
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app_config = Config()

# Cache responses to common queries with small cache (memory constrained)
RESPONSE_CACHE = {}
CACHE_TTL = 300  # 5 minutes
MAX_CACHE_ITEMS = 50  # Limit the number of cached responses


# Cache management function
def manage_cache():
    """Trim cache if it exceeds the maximum number of items"""
    if len(RESPONSE_CACHE) > MAX_CACHE_ITEMS:
        # Remove oldest items
        to_remove = len(RESPONSE_CACHE) - MAX_CACHE_ITEMS
        oldest_keys = sorted(RESPONSE_CACHE.keys(),
                             key=lambda k: RESPONSE_CACHE[k]["time"])[:to_remove]
        for key in oldest_keys:
            del RESPONSE_CACHE[key]
        gc.collect()  # Force garbage collection


# Security dependency updated to bypass OPTIONS requests
async def get_api_key(api_key: str = Security(api_key_header), request: Request = None):
    if request and request.method.upper() == "OPTIONS":
        return ""
    if api_key == app_config.API_KEY:
        return api_key
    logger.warning("Invalid API key attempt")
    raise HTTPException(
        status_code=403,
        detail="Invalid API Key"
    )


# Application state with memory management
class AppState:
    def __init__(self):
        self.config = Config()
        self.chat_service = None
        self.model_loaded_event = asyncio.Event()
        self.loading_lock = asyncio.Lock()
        self.last_gc_time = time.time()
        self.gc_interval = 300  # Run GC every 5 minutes

    async def initialize(self):
        """
        Called once at startup, to create and store a single ChatService instance.
        """
        async with self.loading_lock:
            if not self.chat_service:
                self.chat_service = await asyncio.to_thread(
                    ServiceFactory.create_service, self.config
                )
                self.model_loaded_event.set()
                logger.info("AppState initialized with model: %s", self.config.MODEL_TYPE)
                # Force garbage collection after initialization
                gc.collect()
            return self.chat_service

    async def get_chat_service(self):
        """Get the chat service, waiting for initialization if needed"""
        # Periodic garbage collection
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            self.last_gc_time = current_time
            gc.collect()

        if not self.model_loaded_event.is_set():
            await self.model_loaded_event.wait()
        return self.chat_service

    async def update_model(self, model_type=None, model=None):
        """Update the model asynchronously"""
        async with self.loading_lock:
            self.model_loaded_event.clear()

            # Clean up before loading new model
            old_service = self.chat_service
            self.chat_service = None
            gc.collect()

            # Update config
            if model_type:
                self.config.MODEL_TYPE = model_type

            if model:
                if self.config.MODEL_TYPE == "mlc":
                    self.config.MLC_MODEL = model
                else:
                    self.config.GGUF_MODEL = model

            # Create new service
            self.chat_service = await asyncio.to_thread(
                ServiceFactory.create_service, self.config
            )
            self.model_loaded_event.set()
            logger.info(f"Model updated to {self.config.MODEL_TYPE}")

            # Force garbage collection again
            old_service = None  # Remove reference
            gc.collect()


# Lifespan: set up and tear down the app state
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize app state with model setup
    app.state.app_state = AppState()
    await app.state.app_state.initialize()  # Build service asynchronously
    logger.info("Application started, services initialized")
    yield
    # Shutdown
    logger.info("Application shutting down, cleaning up resources")
    # Force final garbage collection
    gc.collect()


# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Piperag API",
    description="An API for chat and image recognition services.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Error-Type"],
)


# Middleware for request logging with minimal overhead
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = f"{time.time()}-{id(request)}"
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Only log slow requests (>1 second) to reduce logging overhead
        if process_time > 1.0:
            logger.info(f"Slow request: {request.method} {request.url.path} "
                        f"- Status: {response.status_code} - Time: {process_time:.4f}s")

        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} "
                     f"- Error: {str(e)} - Time: {process_time:.4f}s")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(e).__name__}
        )


# Dependencies
async def get_app_state():
    return app.state.app_state


async def get_chat_service(app_state=Depends(get_app_state)) -> ChatService:
    return await app_state.get_chat_service()


def get_config(app_state=Depends(get_app_state)) -> Config:
    return app_state.config


# Streaming response with memory management
@app.get(
    "/ask-stream",
    summary="Streaming Chat Endpoint",
    description="Submit a query to the chat service and receive a streaming response.",
    responses={
        200: {"description": "Successful response", "content": {"text/event-stream": {}}},
        400: {"description": "Bad Request", "model": Dict[str, str]},
        403: {"description": "Forbidden - Invalid API Key", "model": Dict[str, str]},
        500: {"description": "Internal Server Error", "model": Dict[str, str]},
    },
    dependencies=[Depends(get_api_key)]
)
async def ask_stream(
        q: str = Query(..., description="User query", min_length=1),
        model: str = Query(None, description="Optional model identifier"),
        chat_service: ChatService = Depends(get_chat_service),
        config: Config = Depends(get_config)
):
    async def generate():
        try:
            # Stream response chunks
            chunks = await asyncio.to_thread(chat_service.generate_response_stream, q)
            for chunk in chunks:
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Error streaming response", exc_info=True)
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Regular chat endpoint (optimized for memory constraints)
@app.get(
    "/ask",
    summary="Chat Endpoint",
    description="Submit a query to the chat service and receive a response.",
    responses={
        400: {"description": "Bad Request", "model": Dict[str, str]},
        403: {"description": "Forbidden - Invalid API Key", "model": Dict[str, str]},
        500: {"description": "Internal Server Error", "model": Dict[str, str]},
    },
    dependencies=[Depends(get_api_key)]
)
async def ask(
        q: str = Query(..., description="User query", min_length=1),
        model: str = Query(None, description="Optional model identifier"),
        chat_service: ChatService = Depends(get_chat_service),
        config: Config = Depends(get_config),
        app_state: AppState = Depends(get_app_state)
):
    # Check cache for common queries
    cache_key = f"{q}_{model or config.MODEL_TYPE}"
    cached = RESPONSE_CACHE.get(cache_key)
    if cached and (time.time() - cached["time"]) < CACHE_TTL:
        return {"result": cached["result"], "cached": True}

    try:
        model = model.lower() if model else None

        # Handle model switching asynchronously
        if model and model != config.MODEL_TYPE:
            await app_state.update_model(model_type=model)
            chat_service = await app_state.get_chat_service()

        # Generate response using the selected model
        result = await asyncio.to_thread(chat_service.generate_response, q)

        # Cache the result and manage cache size
        RESPONSE_CACHE[cache_key] = {"result": result, "time": time.time()}
        manage_cache()

        return {"result": result}

    except ValueError as e:
        logger.warning(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error getting response", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}",
            headers={"X-Error-Type": type(e).__name__}
        )


# Health check with memory usage stats
@app.get("/health", summary="Health Check", description="Check if the service is running properly")
async def health_check():
    # Get memory usage
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)

    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_type": app.state.app_state.config.MODEL_TYPE,
        "model": app.state.app_state.config.MLC_MODEL if app.state.app_state.config.MODEL_TYPE == "mlc"
        else app.state.app_state.config.GGUF_MODEL,
        "memory_usage_mb": round(memory_mb, 2),
        "cache_items": len(RESPONSE_CACHE)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=2,
        limit_concurrency=50,
        timeout_keep_alive=5,
        log_level="warning",
        loop="uvloop",
        http="httptools",
        access_log=False,
        lifespan="on"
    )