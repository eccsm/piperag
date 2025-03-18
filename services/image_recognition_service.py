import io
import logging
from enum import Enum
from typing import Dict, List, Any, Union
import time

from PIL import Image
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

class RecognitionTask(str, Enum):
    """
    Enum of supported image recognition tasks.
    Using an Enum ensures type safety and documentation.
    """
    DEEPFAKE_DETECTION = "deepfake_detection"
    OBJECT_RECOGNITION = "object_recognition"
    IMAGE_CLASSIFICATION = "image_classification"

class ImageRecognitionService:
    """
    Service for handling various image recognition tasks.
    Supports different models for different tasks through a unified interface.
    """
    
    def __init__(self, config: Config, task: str = None):
        """
        Initialize the image recognition service.
        
        Args:
            config: Application configuration
            task: Recognition task to perform (from RecognitionTask enum)
        
        Raises:
            ValueError: If task is not supported
        """
        self.config = config
        
        # Validate task is supported
        try:
            self.task = RecognitionTask(task) if task else RecognitionTask.DEEPFAKE_DETECTION
        except ValueError:
            valid_tasks = [t.value for t in RecognitionTask]
            raise ValueError(f"Unsupported task: {task}. Must be one of {valid_tasks}")
            
        # Determine hardware device
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")
        
        # Initialize the appropriate model based on task
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model for the selected task."""
        start_time = time.time()
        logger.info(f"Initializing model for task: {self.task}")
        
        try:
            if self.task == RecognitionTask.OBJECT_RECOGNITION:
                # Use the Ultralytics YOLO library for object detection
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")  # Using YOLOv8 instead of YOLOv5
                logger.info("YOLOv8 model loaded for object recognition")
                
            elif self.task == RecognitionTask.IMAGE_CLASSIFICATION:
                # Use transformers pipeline for image classification
                model_name = self.config.IMAGE_MODEL_MAP.get(self.task)
                if not model_name:
                    model_name = self.config.default_image_model
                    
                try:
                    self.model = pipeline("image-classification", model=model_name, device=self.device)
                    logger.info(f"Classification pipeline created with model: {model_name}")
                except Exception as e:
                    # Fallback to manual loading if pipeline fails
                    logger.warning(f"Pipeline creation failed, falling back to manual model loading: {str(e)}")
                    processor = AutoImageProcessor.from_pretrained(model_name)
                    model = AutoModelForImageClassification.from_pretrained(model_name)
                    self.model = pipeline("image-classification", model=model, feature_extractor=processor, device=self.device)
                    
            elif self.task == RecognitionTask.DEEPFAKE_DETECTION:
                # Load deepfake detection model
                model_name = self.config.IMAGE_MODEL_MAP.get(self.task)
                self.model = pipeline("image-classification", model=model_name, device=self.device)
                logger.info(f"Deepfake detection model loaded: {model_name}")
                
            else:
                raise ValueError(f"No implementation for task: {self.task}")
                
            logger.info(f"Model initialization completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing model for {self.task}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def recognize(self, image_data: bytes) -> Dict[str, Any]:
        """
        Recognize content in an image based on the configured task.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Dictionary with recognition results
            
        Raises:
            ValueError: For invalid images
            RuntimeError: For processing errors
        """
        start_time = time.time()
        logger.info(f"Starting recognition task: {self.task}")
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Process based on task type
            if self.task == RecognitionTask.OBJECT_RECOGNITION:
                return self._process_object_detection(image)
            elif self.task == RecognitionTask.IMAGE_CLASSIFICATION:
                return self._process_classification(image)
            elif self.task == RecognitionTask.DEEPFAKE_DETECTION:
                return self._process_deepfake_detection(image)
            else:
                raise ValueError(f"Unsupported task: {self.task}")
                
        except Exception as e:
            logger.error(f"Error in image recognition", exc_info=True)
            raise RuntimeError(f"Image processing failed: {str(e)}")
        finally:
            logger.info(f"Recognition completed in {time.time() - start_time:.2f} seconds")

    def _process_object_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Process image for object detection."""
        # Run YOLOv8 inference
        results = self.model(image)
        
        # Extract detection results - YOLOv8 has a different format than YOLOv5
        detections = []
        
        # Process results - YOLOv8 format is different
        try:
            # Get the first result (only one image)
            result = results[0] if isinstance(results, list) else results
            
            # Convert YOLOv8 results to our standard format
            boxes = result.boxes
            for i, box in enumerate(boxes):
                try:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Get confidence
                    conf = float(box.conf[0])
                    # Get class ID and name
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    
                    detections.append({
                        "label": class_name,
                        "confidence": conf,
                        "bounding_box": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error processing detection {i}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error processing YOLOv8 results: {str(e)}")
            # Just return empty detections if format is unexpected
            pass
            
        return {
            "task": self.task.value,
            "detections": detections,
            "count": len(detections)
        }

    def _process_classification(self, image: Image.Image) -> Dict[str, Any]:
        """Process image for classification."""
        results = self.model(image)
        
        # Extract top classifications
        classifications = []
        for result in results:
            classifications.append({
                "label": result["label"],
                "confidence": float(result["score"])
            })
            
        # Get the top result
        top_result = classifications[0] if classifications else None
        
        return {
            "task": self.task.value,
            "classifications": classifications,
            "top_result": top_result
        }

    def _process_deepfake_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Process image for deepfake detection."""
        results = self.model(image)
        
        # Parse results
        is_fake = False
        confidence = 0.0
        
        for result in results:
            if "fake" in result["label"].lower():
                if result["score"] > confidence:
                    is_fake = True
                    confidence = result["score"]
            elif "real" in result["label"].lower():
                if result["score"] > confidence:
                    is_fake = False
                    confidence = result["score"]
        
        interpretation = "The image appears to be " + ("AI-generated" if is_fake else "authentic")
        
        return {
            "task": self.task.value,
            "is_fake": is_fake,
            "confidence": float(confidence),
            "interpretation": interpretation,
            "full_results": [{"label": r["label"], "confidence": float(r["score"])} for r in results]
        }