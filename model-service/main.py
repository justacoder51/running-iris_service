import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

model = None


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10)
    sepal_width:  float = Field(..., ge=0, le=10)
    petal_length: float = Field(..., ge=0, le=10)
    petal_width:  float = Field(..., ge=0, le=10)


class PredictResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float
    probabilities: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")
    yield
    model = None


app = FastAPI(title="Iris Classifier API", lifespan=lifespan)

# Allow the browser page to call the API from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    features = [[
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width,
    ]]

    probs = model.predict_proba(features)[0]
    class_id = int(probs.argmax())
    label = CLASS_NAMES[class_id]

    return PredictResponse(
        prediction=label,
        class_id=class_id,
        confidence=float(probs[class_id]),
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
