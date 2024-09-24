import os
import base64
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.background import BackgroundTasks
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from PIL import Image
import io
import ssl
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from instance_segmentation import segment_instances
from mask_prediction import predict_mask
from landmarks_detection import detect_landmarks
from evaluation_metrics import f1, iou
from database import SessionLocal, Task, Base
import asyncio
import aiofiles
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

load_dotenv()

# Email configuration
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT")),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS=os.getenv("MAIL_STARTTLS") == 'True',
    MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS") == 'True',
    USE_CREDENTIALS=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

templates = Jinja2Templates(directory="templates")

def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total = K.sum(K.abs(y_true), [1, 2, 3]) + K.sum(K.abs(y_pred), [1, 2, 3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())

    return K.mean(f(y_true, y_pred), axis=-1)

def load_model_with_custom_objects(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'f1': f1, 'iou': iou})
    return model

models = {
    "primary": "models/primary_model",
    "secondary": "models/secondary_model"
}

results: List[Dict[str, Dict]] = []

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict_endpoint(request: Request, file: UploadFile = File(...), selected_model: str = Form(...)):
    try:
        result = await process_image(file, selected_model)
        logger.info("Prediction and landmark detection successful.")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "image": result["image"],
            "landmarks": result["landmarks_html"]
        })
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict/", response_class=HTMLResponse)
async def batch_predict_endpoint(
    request: Request,
    files: List[UploadFile] = File(...),
    selected_model: str = Form(...),
    start_time: str = Form(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        logger.info(f"Received files: {[file.filename for file in files]}")
        logger.info(f"Selected model: {selected_model}")
        logger.info(f"Start time: {start_time}")

        # Adjust start time by subtracting 2 hours to match your local timezone
        start_time_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M") - timedelta(hours=2)
        if start_time_dt < datetime.now():
            raise HTTPException(status_code=400, detail="Start time must be in the future")

        file_paths = []
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            file_path = f"temp/{file.filename}"
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await file.read())
            file_paths.append(file_path)

        task = Task(files=",".join(file_paths), model=selected_model, start_time=start_time_dt)
        db.add(task)
        db.commit()
        db.refresh(task)

        background_tasks.add_task(schedule_batch_processing, task.id)

        return HTMLResponse("<h3>Batch processing scheduled successfully!</h3>")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def schedule_batch_processing(task_id: int):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            logger.error(f"Task with id {task_id} not found.")
            return

        now = datetime.now()
        delay = (task.start_time - now).total_seconds()
        if delay > 0:
            logger.info(f"Waiting for {delay} seconds before starting batch processing")
            await asyncio.sleep(delay)

        file_paths = task.files.split(",")
        for file_path in file_paths:
            await process_and_save_image(file_path, task.model)

        task.status = "completed"
        db.commit()
    finally:
        db.close()

async def process_image(file: UploadFile, model_key: str):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)

    model_path = models.get(model_key)
    if model_key == "user_trained":
        model_path = "models/user_trained_model"

    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Invalid model selected")
    
    model = load_model_with_custom_objects(model_path)

    predicted_mask = predict_mask(model, image_np)

    x, y, w, h = 740, 50, 2804, 2804
    output_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    output_mask[y : y + h, x : x + w] = predicted_mask

    landmarks = detect_landmarks(predicted_mask)
    landmarks_df = pd.DataFrame(landmarks)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np, cmap='gray')
    ax.imshow(output_mask, cmap='jet', alpha=0.5, extent=(0, image_np.shape[1], image_np.shape[0], 0))
    for landmark in landmarks:
        ax.plot(landmark['primary_root_start'][0] + x, landmark['primary_root_start'][1] + y, 'go')
        ax.plot(landmark['primary_root_end'][0] + x, landmark['primary_root_end'][1] + y, 'ro')
        for lrt in landmark['l_root_tips']:
            ax.plot(lrt[0] + x, lrt[1] + y, 'bo')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    result = {
        "file": file.filename,
        "image": image_base64,
        "landmarks_html": landmarks_df.to_html(classes="table table-striped")
    }

    return result

async def process_and_save_image(file_path: str, model_key: str):
    async with aiofiles.open(file_path, 'rb') as f:
        file_data = await f.read()
    file = UploadFile(filename=file_path, file=io.BytesIO(file_data))
    result = await process_image(file, model_key)
    results.append(result)

    result_html = templates.TemplateResponse("result.html", {
        "request": {},
        "image": result["image"],
        "landmarks": result["landmarks_html"]
    })

    async with aiofiles.open(f"results/{os.path.basename(file_path)}.html", "w") as f:
        await f.write(result_html.body.decode('utf-8'))

@app.post("/train_model/", response_class=HTMLResponse)
async def train_model_endpoint(request: Request, file: UploadFile = File(...), epochs: int = Form(...)):
    try:
        print(f"Received file with content type: {file.content_type}")
        
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
        
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))
        image_np = np.array(image) / 255.0
        image_np = np.expand_dims(image_np, axis=0)

        # Define a simple model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(image_np, np.array([1]), epochs=epochs, verbose=0)

        # Save the model in the SavedModel format
        model_path = "models/user_trained_model"
        model.save(model_path, save_format='tf')

        return templates.TemplateResponse("train_result.html", {
            "request": request,
            "accuracy": history.history['accuracy'][-1],
            "loss": history.history['loss'][-1],
            "model_path": model_path
        })
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Middleware to limit requests
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100):
        super().__init__(app)
        self.max_requests = max_requests
        self.requests = {}

    async def dispatch(self, request: StarletteRequest, call_next):
        client_ip = request.client.host
        current_time = datetime.now()

        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Filter out old requests
        self.requests[client_ip] = [
            timestamp for timestamp in self.requests[client_ip]
            if current_time - timestamp < timedelta(minutes=1)
        ]

        # Check if the number of requests exceeds the limit
        if len(self.requests[client_ip]) >= self.max_requests:
            await send_notification_email(f"Too many requests from {client_ip}")
            raise HTTPException(status_code=429, detail="Too many requests")

        self.requests[client_ip].append(current_time)

        response = await call_next(request)
        return response

async def send_notification_email(message: str):
    email_message = MessageSchema(
        subject="Application Alert",
        recipients=[os.getenv("NOTIFICATION_EMAIL")],
        body=message,
        subtype="plain"
    )
    fm = FastMail(conf)
    await fm.send_message(email_message)

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    await send_notification_email(f"HTTP Exception: {exc.detail}")
    return HTMLResponse(content=f"HTTP Exception: {exc.detail}", status_code=exc.status_code)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    await send_notification_email(f"Unhandled Exception: {exc}")
    return HTMLResponse(content="An unexpected error occurred", status_code=500)

app.add_middleware(RateLimitMiddleware, max_requests=100)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
