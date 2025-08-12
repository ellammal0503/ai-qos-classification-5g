#python3 -m venv venv
#source venv/bin/activate

# main.py
import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse

from schema import QoSInputFlat, QoSSequenceInput
from utils import preprocess_flat, preprocess_sequence, preprocess_ip_embedding, load_encoders

from tensorflow.keras.models import load_model

from train_all_models import train_all

app = FastAPI(title="Multi-Model QoS Classifier API")


@app.post("/train")
async def train_all_models(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        result = train_all(file_location)

        os.remove(file_location)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(input_data: dict, model_name: str = Query("random_forest", enum=[
    "random_forest", "svm", "knn", "ip_embed", "lstm", "bilstm"])):
    # ✅ Load encoders including le_qos for label decoding
    #_, _, _, le_qos = load_encoders()

    try:
        if model_name in ["random_forest", "svm", "knn"]:
            parsed = QoSInputFlat(**input_data)
            X = preprocess_flat(parsed)
            model = joblib.load(f"models/{model_name}.pkl")
            pred = model.predict(X)[0]
            # Decode
            _, _, _, le_qos = load_encoders()
            decoded = le_qos.inverse_transform([pred])[0]

            return {"model": model_name, "prediction": int(pred), "decoded_label": decoded}
            #return {"model": model_name, "prediction": int(pred)}

        elif model_name in ["lstm", "bilstm"]:
            parsed = QoSSequenceInput(**input_data)
            X = preprocess_sequence(parsed)
            model = load_model(f"models/{model_name}.h5")
            pred_prob = model.predict(X)
            pred = int(np.argmax(pred_prob))
            _, _, _, le_qos = load_encoders()
            decoded = le_qos.inverse_transform([pred])[0]

            return {"model": model_name, "prediction": pred, "decoded_label": decoded}
            #return {"model": model_name, "prediction": pred}

        elif model_name == "ip_embed":
            parsed = QoSInputFlat(**input_data)
            X = preprocess_ip_embedding(parsed)
            model = load_model("models/ip_embed_model.h5")
            pred_prob = model.predict(X)
            pred = int(np.argmax(pred_prob))
            _, _, _, le_qos = load_encoders()
            decoded = le_qos.inverse_transform([pred])[0]

            return {"model": model_name, "prediction": pred, "decoded_label": decoded}
            #return {"model": model_name, "prediction": pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import HTMLResponse
import os

@app.get("/plots", response_class=HTMLResponse)
async def list_plots():
    metrics_dir = "metrics"
    files = os.listdir(metrics_dir)

    # Group by model name
    models = {}
    for f in files:
        if f.endswith(".png"):
            plot_type, model_name_with_ext = f.split("_", 1)
            model_name = model_name_with_ext.replace(".png", "")
            models.setdefault(model_name, {})[plot_type] = f

    # Build HTML
    html = "<h1>Model Performance Plots</h1>"
    for model_name, plots in models.items():
        html += f"<h2>{model_name}</h2>"
        if "confusion" in plots:
            html += f'<div><strong>Confusion Matrix:</strong><br><img src="/plots/image/{plots["confusion"]}" width="500"></div>'
        if "roc" in plots:
            html += f'<div><strong>ROC Curve:</strong><br><img src="/plots/image/{plots["roc"]}" width="500"></div>'
        html += "<hr>"

    return html

@app.get("/plots/image/{filename}")
async def get_plot_image(filename: str):
    file_path = os.path.join("metrics", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
