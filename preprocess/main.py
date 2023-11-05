from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import io
import uvicorn
from preprocess import preprocess_image
import httpx
# import requests

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API OK!"}

@app.post("/api/preprocess/")
async def upload_file(file: UploadFile):
    if file.content_type.startswith('image/'):
        img_data = await file.read()
        img_array = preprocess_image(io.BytesIO(img_data))
        if img_array is not None:
            processed_image = img_array.tolist()

    processed_data = {
    "file_name": file.filename,
    "processed_image": processed_image
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post('http://0.0.0.0:5000/api/predict/', json=processed_data)
   
    if response.status_code == 200:
        return JSONResponse(content=response.json())
    else:
        print(processed_data)
        return JSONResponse(content={"error": "Failed to send data to another API"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
