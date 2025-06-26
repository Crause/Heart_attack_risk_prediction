from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import argparse
import logging
import io
from predictor import Predictor

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/input-example")
def load_template():
    filename = 'input_example.csv'
    return FileResponse(f'templates\{filename}', media_type='text/csv', filename=filename)

@app.post("/predict")
async def predict(upload_file: UploadFile = File(...)):
    try:
        predictor = Predictor()
        predictor.predict(upload_file.file)
        
        stream = io.StringIO()
        predictor.predictions.to_csv(stream)
        stream.seek(0)
        predictor.__fill_missing
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=pred_{upload_file.filename}"}
        )
    finally:
        upload_file.file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())
    
    uvicorn.run(app, **args)