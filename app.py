import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from PIL import Image
import io
import os
import cloudinary
import cloudinary.uploader

# Initialize the FastAPI app
app = FastAPI()

cloudinary.config(
    cloud_name="******",
    api_key="*****",
    api_secret="********",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the YOLO model
model = YOLO("./best1.pt")  # Replace with the path to your model file

# Temporary folder to store downloaded images and results
TEMP_FOLDER = "./temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)


@app.post("/predict/")
async def predict(image_url: str):
    try:
        # Step 1: Download the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")

        image_data = io.BytesIO(response.content)
        image = Image.open(image_data)
        input_image_path = os.path.join(TEMP_FOLDER, "input_image.jpg")
        image.save(input_image_path)

        # Step 2: Run inference using the YOLO model
        results = model([input_image_path])

        # Step 3: Save the output image
        output_image_path = os.path.join(TEMP_FOLDER, "output_image.jpg")
        results[0].save(filename=output_image_path)

        # Step 4: Upload the output image to Cloudinary
        cloudinary_response = cloudinary.uploader.upload(output_image_path)
        cloudinary_url = cloudinary_response.get("secure_url")

        # Step 5: Return the Cloudinary URL and save the image locally
        return JSONResponse(content={"cloudinary_url": cloudinary_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
