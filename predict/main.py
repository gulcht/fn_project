from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from predict import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API OK!"}

@app.post('/api/predict/')
async def process_received_data(request: Request):
    try:
        data = await request.json()

        if "processed_image" in data:
            processed_image = data["processed_image"]
            file_name = data["file_name"]
            predicted_class = predict(processed_image)

        predicted_data = {
            "file_name": file_name,
            "predicted_class": predicted_class
            }
        
        return JSONResponse(content=predicted_data)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


