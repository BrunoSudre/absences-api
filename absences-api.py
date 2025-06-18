import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Absences API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


class AbsencePredictionRequest(BaseModel):
    ProposedAssignmentDurationHours: int
    Gender: int
    AgeAtAssignment: int
    TenureDaysAtAssignment: int
    IsBirthdayOnTargetDate: bool
    HasVehicule: bool
    EmployeeMaxTravelDistancePreference: int
    AllowOvertime: bool
    AllowEveningShift: bool
    AllowNightShift: bool
    HourlyRate: int
    EmployeeType: int
    NumberOfSkills: int
    TravelDistanceForAssignment: int
    PastCallOffs_Last90Days_Prior: int
    DayOfWeek: int
    Month: int
    DayOfMonth: int
    IsWeekend: bool

class AbsencePredictionResponse(BaseModel):
    absence_prediction: float

class AbsenceRetrainingRequest(AbsencePredictionRequest):
    IsAbsent: bool

class MessageResponse(BaseModel):
    message: str

@app.post("/predict-absences")
def predict_absences(data: AbsencePredictionRequest = None):
    pipe = pickle.load(open("absences_sgdc_pipe.pkl", "rb"))
    input_df = pd.DataFrame([data.model_dump()])
    print("Input DataFrame for prediction:", input_df.columns.tolist())
    prediction = pipe.predict_proba(input_df)[0][1]  # get the probability of absence
    print("Prediction probability:", prediction)
    return AbsencePredictionResponse(absence_prediction=prediction)


@app.post("/retrain")
def retrain_model(retraining_data: AbsenceRetrainingRequest):
    model_file_name = "absences_sgdc_pipe.pkl"
    pipe = pickle.load(open(model_file_name, "rb"))

    print(retraining_data.model_dump())
    X = pd.DataFrame([retraining_data.model_dump()]).drop(columns=["IsAbsent"])
    y = [1] if retraining_data.IsAbsent else [0]

    # update the pipeline file with the retrained model
    print("Retraining model with new data:", X.columns.tolist(), "and label:", y)

    scaler = pipe.named_steps["scaler"]
    classifier = pipe.named_steps["classifier"]

    # applies the scaler to the new data
    X_scaled = scaler.transform(X)

    # applies the classifier to the transformed data
    classifier.partial_fit(X_scaled, y)

    # update the pipeline with the retrained classifier
    pipe.named_steps["classifier"] = classifier

    print(f"Model retrained successfully, saving to file [{model_file_name}]...")
    with open(model_file_name, "wb") as f:
        pickle.dump(pipe, f)

    return MessageResponse(message="Model retrained successfully")