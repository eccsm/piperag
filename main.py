import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from typing import Union, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, BackgroundTasks, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from jinja2 import Environment, FileSystemLoader
from starlette.responses import StreamingResponse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from services.image_recognition_service import ImageRecognitionService
from services.service_factory import ServiceFactory
from services.chat_service import ChatService
from config import Config

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("piperag")

# Define API key security scheme
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize config to get API key from .env via Config class
app_config = Config()


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


# Application state
class AppState:
    def __init__(self):
        self.config = Config()
        self.chat_service = None  # This will hold our ChatService instance

    def initialize(self):
        """
        Called once at startup, to create and store a single ChatService instance.
        """
        self.chat_service = ServiceFactory.create_service(self.config)
        logger.info("AppState initialized with model: %s", self.config.MODEL_TYPE)


# Lifespan: set up and tear down the app state
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize app state with model setup
    app.state.app_state = AppState()
    app.state.app_state.initialize()  # Build service once at startup
    logger.info("Application started, services initialized")
    yield
    # Shutdown
    logger.info("Application shutting down, cleaning up resources")


# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Piperag API",
    description="An API for chat and image recognition services.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = ["*"]
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Error-Type"],
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = f"{time.time()}-{id(request)}"
    logger.info(f"Request started: {request.method} {request.url.path} (ID: {request_id})")
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.url.path} "
                    f"(ID: {request_id}) - Status: {response.status_code} - Time: {process_time:.4f}s")
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} "
                     f"(ID: {request_id}) - Error: {str(e)} - Time: {process_time:.4f}s")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(e).__name__}
        )


RESUME_PATH = "/app/resume.json"


# Dependencies
def get_app_state():
    return app.state.app_state


def get_chat_service(app_state=Depends(get_app_state)) -> ChatService:
    return app_state.chat_service


def get_config(app_state=Depends(get_app_state)) -> Config:
    return app_state.config


def load_resume():
    with open(RESUME_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Common error responses
responses = {
    400: {"description": "Bad Request", "model": Dict[str, str]},
    403: {"description": "Forbidden - Invalid API Key", "model": Dict[str, str]},
    500: {"description": "Internal Server Error", "model": Dict[str, str]},
}


@app.get(
    "/ask",
    summary="Chat Endpoint",
    description="Submit a query to the chat service and receive a response.",
    responses=responses,
    dependencies=[Depends(get_api_key)]
)
async def ask(
        q: str = Query(..., description="User query", min_length=1),
        model: str = Query(None, description="Optional model identifier"),
        chat_service: ChatService = Depends(get_chat_service),
        config: Config = Depends(get_config)
):
    try:
        model = model.lower() if model else None
        old_model_type = config.MODEL_TYPE
        model_changed = False

        # Handle model type changes based on shorthand model names
        if model in ["sug", "pep"]:
            # Update cat personality
            config.CAT_NAME_EN = model.capitalize()
            name_map = {"sug": "Şeker", "pep": "Biber"}
            config.CAT_NAME_TR = name_map[model]

            # Switch to GGUF and select the appropriate model
            if config.MODEL_TYPE != "gguf":
                config.MODEL_TYPE = "gguf"
                model_changed = True

            # Set the GGUF model to match the cat personality
            config.GGUF_MODEL = model

        elif model == "mlc":
            # Switch to MLC model type
            if config.MODEL_TYPE != "mlc":
                config.MODEL_TYPE = "mlc"
                model_changed = True
        elif model:
            # Handle custom model selection
            if model in config.get_mlc_model_map():
                config.MODEL_TYPE = "mlc"
                config.MLC_MODEL = model
                model_changed = True
            elif model in config.get_gguf_model_map():
                config.MODEL_TYPE = "gguf"
                config.GGUF_MODEL = model
                model_changed = True

        # Reinitialize service if model type changed
        if model_changed or config.MODEL_TYPE != old_model_type:
            logger.info(f"Switching model from {old_model_type} to {config.MODEL_TYPE}")
            app.state.app_state.chat_service = ServiceFactory.create_service(config)
            chat_service = app.state.app_state.chat_service
        else:
            logger.debug("Model unchanged, skipping re-init.")

        # Generate response using the selected model
        result = await asyncio.to_thread(chat_service.generate_response, q)
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


@app.post(
    "/update_model",
    summary="Update Model",
    description="Update the backend model configuration.",
    responses=responses,
    dependencies=[Depends(get_api_key)]
)
async def update_model(
        background_tasks: BackgroundTasks,
        new_model: str = Body("", embed=True, description="Model name, path or HF URL"),
        new_model_type: Union[str, None] = Body(None, embed=True,
                                                description="Model type (e.g., 'sug', 'pep', 'mlc', or a model key)"),
        config: Config = Depends(get_config),
        chat_service: ChatService = Depends(get_chat_service),
):
    try:
        old_model_type = config.MODEL_TYPE
        model_changed = False

        # Handle model type changes
        if new_model_type:
            new_model_type = new_model_type.lower()

            # Handle predefined personalities
            if new_model_type in ["sug", "pep"]:
                # Update cat personality
                config.CAT_NAME_EN = new_model_type.capitalize()
                name_map = {"sug": "Şeker", "pep": "Biber"}
                config.CAT_NAME_TR = name_map[new_model_type]

                # Switch to GGUF
                config.MODEL_TYPE = "gguf"
                config.GGUF_MODEL = new_model_type
                model_changed = True

            # Handle direct model type specification
            elif new_model_type == "mlc":
                config.MODEL_TYPE = "mlc"
                model_changed = True
            elif new_model_type == "gguf":
                config.MODEL_TYPE = "gguf"
                model_changed = True

            # Handle model selection by key
            elif new_model_type in config.get_mlc_model_map():
                config.MODEL_TYPE = "mlc"
                config.MLC_MODEL = new_model_type
                model_changed = True
            elif new_model_type in config.get_gguf_model_map():
                config.MODEL_TYPE = "gguf"
                config.GGUF_MODEL = new_model_type
                model_changed = True
            else:
                raise ValueError(f"Unsupported model type: {new_model_type}")

        # Handle direct model specification
        if new_model and new_model.strip():
            if config.MODEL_TYPE == "mlc":
                # For MLC, handle both HF:// URLs and direct paths
                config.MLC_MODEL = new_model
                model_changed = True
            elif config.MODEL_TYPE == "gguf":
                # For GGUF, set the model name/path
                config.GGUF_MODEL = new_model
                model_changed = True

        # Reinitialize service if model changed
        if model_changed or config.MODEL_TYPE != old_model_type:
            logger.info(f"Switching model from {old_model_type} to {config.MODEL_TYPE}")
            app.state.app_state.chat_service = ServiceFactory.create_service(config)
        else:
            logger.debug("Model unchanged, skipping re-init.")

        # Return the current configuration
        return {
            "status": "success",
            "message": "Model updated successfully",
            "model_type": config.MODEL_TYPE,
            "model": config.MLC_MODEL if config.MODEL_TYPE == "mlc" else config.GGUF_MODEL
        }
    except ValueError as e:
        logger.warning(f"Invalid model configuration: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error updating model", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update model: {str(e)}",
            headers={"X-Error-Type": type(e).__name__}
        )


@app.post(
    "/recognize",
    summary="Image Recognition",
    description="Perform image recognition using the specified task.",
    responses=responses,
    dependencies=[Depends(get_api_key)]
)
async def recognize(
        task: str = Query(..., description="Task for image recognition"),
        file: UploadFile = File(..., description="Image file to analyze"),
        config: Config = Depends(get_config)
):
    valid_tasks = list(config.IMAGE_MODEL_MAP.keys())
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {task}. Must be one of {valid_tasks}"
        )

    content_type = file.content_type
    if not content_type or not content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got content type: {content_type}"
        )

    try:
        contents = await file.read()
        image_service = ImageRecognitionService(config, task=task)
        result = await asyncio.to_thread(image_service.recognize, contents)
        return {"result": result}
    except ValueError as e:
        logger.warning(f"Invalid input for recognition: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}",
            headers={"X-Error-Type": type(e).__name__}
        )


@app.get("/health", summary="Health Check", description="Check if the service is running properly")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_type": app.state.app_state.config.MODEL_TYPE,
        "model": app.state.app_state.config.MLC_MODEL if app.state.app_state.config.MODEL_TYPE == "mlc"
        else app.state.app_state.config.GGUF_MODEL
    }


@app.get("/models", summary="List Available Models", description="Get a list of available models")
async def list_models(config: Config = Depends(get_config)):
    return {
        "current_type": config.MODEL_TYPE,
        "current_model": config.MLC_MODEL if config.MODEL_TYPE == "mlc" else config.GGUF_MODEL,
        "available_models": {
            "mlc": config.get_mlc_model_map(),
            "gguf": config.get_gguf_model_map(),
        }
    }


env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("resume_template.html")


@app.get("/generate_resume_pdf")
async def generate_resume_pdf():
    data = load_resume()
    rendered_html = template.render(data)

    temp_html_file = f"temp_{uuid.uuid4().hex}.html"
    with open(temp_html_file, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    # Generate a temporary PDF file name.
    temp_pdf_file = f"temp_{uuid.uuid4().hex}.pdf"

    # Run Pandoc with xelatex.
    pandoc_cmd = [
        "pandoc",
        temp_html_file,
        "-o",
        temp_pdf_file,
        "--pdf-engine=xelatex"
    ]

    try:
        subprocess.run(pandoc_cmd, check=True)
    except subprocess.CalledProcessError:
        os.remove(temp_html_file)
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    os.remove(temp_html_file)  # Clean up the HTML file.

    def iterfile():
        with open(temp_pdf_file, "rb") as pdf_file:
            yield pdf_file.read()
        os.remove(temp_pdf_file)

    return StreamingResponse(
        iterfile(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume.pdf"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)