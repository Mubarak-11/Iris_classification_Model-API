import os, logging,time, joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
app = FastAPI(title= "Iris Classification Model API", version="1.0")


#import schema classes
from server.schemas import( predictRequest, predictIrisResponse)

#import model_def methods
from server.model_def import load_iris_weights, forward_classify, warmup


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

MODEL_IRIS_VER = os.getenv("MODEL_IRIS_VER", "v1")

@app.on_event("startup")
def _startup():
    
    global model

    #first lets load the models
    logger.info("Loading Models....")

    weights_path = os.getenv("MODEL_IRIS_PATH", "iris_model_weights.pkl")

    #load the model and such now
    try:
        model = load_iris_weights(weights_path, in_features=4) #features equals 4

        #warmup to catch issues
        warmup(model, n_features=4)
        app.state.mode_iris = model

        logger.info("Iris model successfully loaded on Startup!")
    
    except Exception as e:
        logger.info(f" Failed to load iris model: {str(e)}")
        raise

@app.get("/debug/files", include_in_schema=False)
def debug_files():
    return {"files": os.listdir("/models")}

@app.get("/health")
def health():
    ok = all(os.getenv("MODEL_IRIS_PATH", "iris_model_weights.pkl"))
    return {"ok": bool(ok), "model_version": MODEL_IRIS_VER}


@app.post("/predict", response_model= predictIrisResponse)
def predict(req: predictRequest):
    
    try:
        x = np.array([[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]], dtype=np.float32)
        #x_t = torch.from_numpy(x)

        pred, prob = forward_classify(model, x)
        
        return {"predicted_class": pred,"probabilities": prob,"model_version": MODEL_IRIS_VER}
     
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Classification Error: {e}")