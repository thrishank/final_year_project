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

app = FastAPI()

cloudinary.config(
    cloud_name="******",
    api_key="*****",
    api_secret="********",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],  
)
 
model = YOLO("./best1.pt")  

TEMP_FOLDER = "./temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)


@app.post("/predict/")
async def predict(image_url: str):
    try:
         
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")

        image_data = io.BytesIO(response.content)
        image = Image.open(image_data)
        input_image_path = os.path.join(TEMP_FOLDER, "input_image.jpg")
        image.save(input_image_path)
         
        results = model([input_image_path])
        
        output_image_path = os.path.join(TEMP_FOLDER, "output_image.jpg")
        results[0].save(filename=output_image_path)
        
        cloudinary_response = cloudinary.uploader.upload(output_image_path)
        cloudinary_url = cloudinary_response.get("secure_url")

        return JSONResponse(content={"cloudinary_url": cloudinary_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
