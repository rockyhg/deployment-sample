import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image, UnidentifiedImageError

from predict import predict


app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Please use POST images"}


@app.post("/predict")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()

        try:
            with Image.open(io.BytesIO(content)) as image:
                result = predict(image)
                print(f'result = {result}')

        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        return JSONResponse(content={"result": result})

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        await file.close()  # ファイルリソースを解放
