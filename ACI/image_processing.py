import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from fastapi.templating import Jinja2Templates

from instance_segmentation import segment_instances
from mask_prediction import predict_mask
from landmarks_detection import detect_landmarks
from evaluation_metrics import f1, iou

templates = Jinja2Templates(directory="templates")

def load_model_with_custom_objects(model_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, custom_objects={'f1': f1, 'iou': iou})
    return model

models = {
    "primary": "models/primary_model",
    "secondary": "models/secondary_model"
}

async def process_image(file_path: str, model_key: str):
    image = Image.open(file_path).convert("RGB")
    image_np = np.array(image)

    # Load the selected model
    model_path = models.get(model_key)
    if not model_path:
        raise ValueError("Invalid model selected")
    model = load_model_with_custom_objects(model_path)

    # Perform prediction
    predicted_mask = predict_mask(model, image_np)

    # Create the output mask
    x, y, w, h = 740, 50, 2804, 2804
    output_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    output_mask[y : y + h, x : x + w] = predicted_mask

    # Perform landmark detection
    landmarks = detect_landmarks(predicted_mask)

    # Convert landmarks to DataFrame for tabular format
    landmarks_df = pd.DataFrame(landmarks)

    # Overlay mask on the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np, cmap='gray')
    ax.imshow(output_mask, cmap='jet', alpha=0.5, extent=(0, image_np.shape[1], image_np.shape[0], 0))
    for landmark in landmarks:
        ax.plot(landmark['primary_root_start'][0] + x, landmark['primary_root_start'][1] + y, 'go')  # start point
        ax.plot(landmark['primary_root_end'][0] + x, landmark['primary_root_end'][1] + y, 'ro')  # end point
        for lrt in landmark['l_root_tips']:
            ax.plot(lrt[0] + x, lrt[1] + y, 'bo')  # lateral root tips
    plt.axis('off')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    result = {
        "file": file_path,
        "image": image_base64,
        "landmarks_html": landmarks_df.to_html(classes="table table-striped")
    }

    return result

async def process_and_save_image(file_path: str, model_key: str):
    result = await process_image(file_path, model_key)
    
    # Save result in HTML template
    result_html = templates.TemplateResponse("result.html", {
        "request": {},
        "image": result["image"],
        "landmarks": result["landmarks_html"]
    })

    result_path = f"results/{os.path.basename(file_path)}.html"
    with open(result_path, "w") as f:
        f.write(result_html.body.decode('utf-8'))

    return {"file_path": file_path, "result_path": result_path}
