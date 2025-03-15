import os
import torch
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from transformers import AutoProcessor, AutoModel, AutoTokenizer, BertTokenizer, BertForQuestionAnswering

# === Load Flask App === #
app = Flask(__name__)

print('✅ Model loading...')

# === Load VGG19 for Brain Tumor Classification === #
MODEL_VGG19_PATH = r'C:\Users\Vishnu\annabrainhac\model_weights\vgg_unfrozen.h5'
VGG19_WEIGHTS_PATH = r'C:\Users\Vishnu\annabrainhac\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG19(include_top=False, input_shape=(240, 240, 3), weights=VGG19_WEIGHTS_PATH)
x = Flatten()(base_model.output)
x = Dense(4608, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1152, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model_03 = Model(inputs=base_model.input, outputs=output)
model_03.load_weights(MODEL_VGG19_PATH)

# === Load BioViL-T Model for Image Analysis === #
MODEL_VLM_PATH = r"C:\Users\Vishnu\annabrainhac\vlm_model"
processor = AutoProcessor.from_pretrained(MODEL_VLM_PATH, trust_remote_code=True)
vlm_model = AutoModel.from_pretrained(MODEL_VLM_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === Load BERT for Question Answering === #
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
bert_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

print('✅ Models loaded successfully. Check http://127.0.0.1:5000/')

# === Helper Functions === #
def get_class_name(class_no):
    return "Yes, Brain Tumor Detected" if class_no == 1 else "No Brain Tumor Detected"

def get_result(img_path):
    """Predict brain tumor using VGG19"""
    try:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).resize((240, 240))
        image = np.expand_dims(np.array(image), axis=0)
        image = preprocess_input(image)
        prediction = model_03.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]
        return get_class_name(class_index), prediction[0]
    except Exception as e:
        return f"Error processing image: {e}", None

def analyze_image_with_vlm(img_path, question=None):
    """Analyze MRI scan using BioViL-T model"""
    try:
        image = Image.open(img_path).convert("RGB")
        text_prompt = question if question else "Analyze this brain MRI scan for abnormalities."
        image_inputs = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
        text_inputs = tokenizer(text_prompt, return_tensors="pt")["input_ids"].to("cuda")
        vlm_model.to("cuda")
        outputs = vlm_model(pixel_values=image_inputs, input_ids=text_inputs)
        return f"✅ Image analysis complete. Feature vector shape: {outputs.last_hidden_state.shape}"
    except Exception as e:
        return f"❌ Error analyzing image: {e}"

def ai_agent_advice(prediction):
    """Generate AI-based advice based on tumor probability"""
    if prediction is None:
        return "AI analysis failed. No suggestions available."
    brain_tumor_prob = prediction[1]
    if brain_tumor_prob > 0.75:
        return "The model strongly suggests the presence of a brain tumor. Consult a neurologist for further MRI scans and clinical evaluation."
    elif 0.50 < brain_tumor_prob <= 0.75:
        return "Moderate chance of a tumor. Follow-up with a specialist is advised."
    else:
        return "No significant signs of a tumor detected. If symptoms persist, consult a doctor."

def detailed_ai_suggestions(prediction):
    """Provide detailed AI-based risk analysis"""
    if prediction is None:
        return "Unable to generate AI-based suggestions due to missing prediction data."
    brain_tumor_prob = prediction[1]
    if brain_tumor_prob > 0.90:
        return "🔴 High-risk category. Immediate medical attention is required."
    elif 0.75 < brain_tumor_prob <= 0.90:
        return "🟠 Medium-High risk. Further MRI and biopsy recommended."
    elif 0.50 < brain_tumor_prob <= 0.75:
        return "🟡 Moderate risk. Follow-up MRI in a few months suggested."
    else:
        return "🟢 Low risk. No immediate concerns."

def answer_question(question, context):
    """Generate an answer using BERT for Q&A"""
    inputs = bert_tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer if answer else "I couldn't find an answer."

# === Flask Routes === #
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    """Handle image upload and tumor detection"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    try:
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(file.filename))
        file.save(file_path)

        classification_result, prediction = get_result(file_path)
        vlm_result = analyze_image_with_vlm(file_path)
        ai_advice = ai_agent_advice(prediction)
        detailed_advice = detailed_ai_suggestions(prediction)

        return jsonify({
            "classification_result": classification_result,
           
            "ai_advice": ai_advice,
            "detailed_risk_analysis": detailed_advice
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions using BERT"""
    try:
        data = request.json
        question = data.get("question")
        context = data.get("context", "This is an AI-powered brain tumor detection system.")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        answer = answer_question(question, context)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"Failed to process question: {e}"}), 500

# === Run App === #
if __name__ == '__main__':
    app.run(debug=True)
