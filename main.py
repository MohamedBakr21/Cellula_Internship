import io
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# Import your custom modules
from week_two.mongo_db.database import save_prediction
from week_two.image_captioning.model import ImageCaptioner

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Initialize Models Globally (Prevents reloading on every request) ---
CURRENT_PATH = Path(__file__).resolve().parent
adapter_path = CURRENT_PATH / "week_one/Bonus/my_saved_model/toxic_classifier"
tokenizer_path = CURRENT_PATH / "week_one/Bonus/my_saved_model/toxic_tokenizer"

# Load Toxicity Classifier
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
config = PeftConfig.from_pretrained(adapter_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path, 
    num_labels=9
)
toxic_model = PeftModel.from_pretrained(base_model, adapter_path)
toxic_model.eval()

# Initialize Image Captioner Instance
captioner_instance = ImageCaptioner()

id2label = {
    0: 'Child Sexual Exploitation', 1: 'Elections', 2: 'Non-Violent Crimes', 
    3: 'Safe', 4: 'Sex-Related Crimes', 5: 'Suicide & Self-Harm', 
    6: 'Unknown S-Type', 7: 'Violent Crimes', 8: 'unsafe'
}

@app.post("/process")
async def process_request(text: str = Form(""), file: UploadFile = File(None)):
    image_caption = ""
    
    # Step A: Handle Image if uploaded
    if file:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            # Call the method on the instance, not the class
            image_caption = captioner_instance.caption(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    # Step B: Determine what to classify
    # If image exists, classify its caption. Otherwise, use user text.
    inference_text = image_caption if image_caption else text
    
    if not inference_text:
        return {"result": "No input provided", "processed_text": "Empty"}

    # Step C: Run Toxicity Inference
    inputs = tokenizer(inference_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = toxic_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    score, class_id = torch.max(probs, dim=-1)
    label = id2label[class_id.item()]
    confidence = float(score.item())

    # Step D: Save to MongoDB
    save_prediction(
        user_text=text,
        image_caption=image_caption,
        label=label,
        score=confidence,
        model_name="DistilBert-LoRA-Safety"
    )

    return {
        "processed_text": inference_text,
        "result": label,
        "score": f"{confidence:.2%}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)