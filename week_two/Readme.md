# 🛡️ AI Safety Auditor

A multimodal content moderation system that uses **Natural Language Processing (NLP)** and **Computer Vision** to detect toxic content. The system accepts text or images, generates captions for images using a BLIP model, and classifies the resulting text into one of nine safety categories using a DistilBERT model optimized with LoRA.

## 🚀 Features

* **Multimodal Input**: Processes raw text or uploaded images.
* **Image Captioning**: Automatically describes images using the `Salesforce/blip-image-captioning-base` model.
* **Safety Classification**: Categorizes content into 9 specific types (e.g., Safe, Violent Crimes, Elections).
* **LoRA Optimization**: Utilizes a Parameter-Efficient Fine-Tuned (PEFT) model for high accuracy with lower memory overhead.
* **Persistent Storage**: Automatically logs all user inputs, generated captions, and model predictions to **MongoDB**.
* **Modern Web UI**: A clean, single-page interface with real-time feedback.

---

## 🏗️ Project Architecture

The system is built with a decoupled architecture to ensure scalability and clean code management:

1. **Frontend**: HTML5/CSS3/JavaScript (Fetch API).
2. **API Layer**: FastAPI (Python).
3. **Inference Engines**:
* **Vision**: BLIP (Hugging Face Transformers).
* **Text**: DistilBERT + LoRA (PEFT/Transformers).


4. **Database**: MongoDB.

---

## 📋 Prerequisites

Ensure you have the following installed:

* Python 3.9+
* MongoDB (running on `localhost:27017`)
* CUDA Toolkit (Optional, for GPU acceleration)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Cellula_Internship

```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn transformers torch peft pillow python-multipart pymongo

```

### 3. Folder Structure

Ensure your saved models are in the following paths as expected by `main.py`:

```text
week_one/Bonus/my_saved_model/
├── toxic_classifier/  # LoRA adapter weights & config
└── toxic_tokenizer/   # Tokenizer files

```

### 4. Run the Application

Start the FastAPI server:

```bash
python main.py

```

The server will start at `http://127.0.0.1:8000`.

---

## 📂 API Reference

### Process Content

**Endpoint:** `POST /process`

**Parameters (Form Data):**
| Key | Type | Description |
| :--- | :--- | :--- |
| `text` | String | Optional. The text to be analyzed. |
| `file` | File | Optional. Image file to be captioned and analyzed. |

**Response:**

```json
{
  "processed_text": "A description of the uploaded image",
  "result": "Safe",
  "score": "98.50%"
}

```

---

## 🏷️ Label Mapping

The model classifies content into the following categories:

* `0`: Child Sexual Exploitation
* `1`: Elections
* `2`: Non-Violent Crimes
* `3`: Safe
* `4`: Sex-Related Crimes
* `5`: Suicide & Self-Harm
* `6`: Unknown S-Type
* `7`: Violent Crimes
* `8`: Unsafe
