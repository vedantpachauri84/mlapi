from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from predictor import predict_employment


app = FastAPI(
    title="SkillPredict AI",
    description="Reusable ML API for employment readiness and skill-gap prediction.",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class TraineeData(BaseModel):
    attendance: float = Field(ge=0, le=100)
    assessment: float = Field(ge=0, le=100)
    practical: float = Field(ge=0, le=100)
    completion: float = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=1)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "SkillPredict AI",
        "version": "1.0"
    }


@app.post("/predict")
def predict(data: TraineeData):

    result = predict_employment(
        attendance=data.attendance,
        assessment=data.assessment,
        practical=data.practical,
        completion=data.completion,
        experience=data.experience,
    )

    return result