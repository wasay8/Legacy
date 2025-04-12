import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer from Hugging Face (ensure no device-specific movement)
model = BertForSequenceClassification.from_pretrained('wasay8/bert-mental-health-lq-hq-mq')
tokenizer = BertTokenizer.from_pretrained('wasay8/bert-mental-health-lq-hq-mq')

# App config
st.set_page_config(page_title="🧠 Mental Health Classifier", layout="centered")

# App header
st.markdown(
    """
    <div style="text-align:center; padding: 10px;">
        <h1 style="color:#4B8BBE;">🧠 Mental Health Response Classifier</h1>
        <p style="font-size:18px; color:#444;">Enter a prompt and a response to see how well the response aligns in quality.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Section
with st.container():
    st.subheader("📝 Input Fields")
    prompt = st.text_area(
        "🗣️ Prompt",
        "I would do about anything to lose these 50lbs. But dieting doesn't work for me. Could you tell me what drugs or even surgery might help?",
        help="Enter a mental health-related prompt."
    )
    response = st.text_area(
        "💬 Response",
        "You don't know what else to do.",
        help="Enter the response to the prompt."
    )

# Classification logic
def classify_quality(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Check if inputs are on the CPU
    if any(input_tensor.device != torch.device('cpu') for input_tensor in inputs.values()):
        inputs = {k: v.to('cpu') for k, v in inputs.items()}

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Ensure that logits are valid tensors
    logits = outputs.logits
    if logits is None:
        raise ValueError("Model did not return valid logits.")
    
    # Check if logits are in an expected state before getting prediction
    if logits.is_meta:
        raise RuntimeError("Model output contains meta tensors, which are not valid for predictions.")

    # Get the prediction from the logits
    prediction = torch.argmax(logits, dim=-1)
    
    # Ensure it's a valid tensor and then get the value
    if prediction.numel() > 0:
        return prediction.item()
    else:
        raise ValueError("Prediction tensor is empty.")

def display_quality(prediction):
    label_map = {
        0: ("Low Quality", "#FF4B4B", "⚠️"),
        1: ("High Quality", "#2ECC71", "✅"),
        2: ("Mid Quality", "#F1C40F", "🟡"),
    }
    if prediction is not None:
        label, color, icon = label_map[prediction]
        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 15px; border-radius: 10px; text-align: center;">
                <h2 style="color:white;">{icon} Predicted Response Quality: {label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# Action button
st.markdown("---")
st.markdown("### 🔍 Run Classification")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_button = st.button("🚀 Classify Now", use_container_width=True)

if classify_button:
    if prompt and response:
        input_text = f"Prompt: {prompt.strip()}\nResponse: {response.strip()}"
        with st.spinner("Analyzing response quality..."):
            try:
                prediction = classify_quality(input_text)
                display_quality(prediction)
            except Exception as e:
                st.error(f"❌ Error during classification: {e}")
    else:
        st.warning("⚠️ Please fill in both the prompt and the response fields before classifying.")
