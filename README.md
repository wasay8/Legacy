# 🧠 Mental Health Counseling Assistant

This project is a **Streamlit web application** designed to assist mental health professionals. It provides two main functionalities:

1. **Information Retrieval & Suggestions using LLM (RAG)**
2. **Response Quality Classification using Machine Learning**

---

## Structure

- `Project/Welcome.py`: Landing page with navigation guidance.
- `Project/pages/Database_and_LLM.py`: Implements Retrieval-Augmented Generation (RAG) and direct LLM suggestions.
- `Project/pages/Machine_learning.py`: Predicts the quality of mental health counseling responses.
- `Project/requirements.txt`: Install the required libraries to execute.
- `src/finetune.ipynb`: Fine tuning of Bert model for Machine Learning Prediction task of mental health counseling.


---

## Features

### Task 1: Search & Suggest with LLM (RAG)

- Upload datasets (`counselchat-data.csv`, `train.csv`).
- Automatically embeds documents into a vector database using FAISS.
- Ask questions and get context-aware answers from the LLM.
- Alternatively, get direct suggestions from LLM without retrieval.

### Task 2: ML-Based Response Quality Classification

- Input a prompt and a mental health response.
- Predicts whether the response is:
  - ✅ High Quality
  - 🟡 Mid Quality
  - ⚠️ Low Quality

---

## Getting Started

### 1. **Clone the repository**

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. **Install requirements**

```bash
pip install -r requirements.txt
```

### 3. **Set up your environment**

- Create a `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY_2=your-openai-api-key
```

- Make sure your folder structure is like:

```
Legacy/
├── src/
│   ├── finetune.ipynb
├── Project/
│   ├── data/
│   │   ├── counselchat-data.csv
│   │   └── train.csv
│   ├── pages/
│   │   ├── Database_and_LLM.py
│   │   └── Machine_learning.py
│   ├── Welcome.py
│   ├── requirements.txt


```

### 4. **Run the Streamlit App**

```bash
streamlit run Welcome.py
```

---

## Tech Stack

- Python
- Streamlit
- OpenAI (GPT-4)
- LangChain
- FAISS
- HuggingFace Transformers

---

## Notes

- The LLM assistant uses GPT-4 for both retrieval and direct suggestions.
- The classifier uses a custom fine-tuned BERT model: `wasay8/bert-mental-health-lq-hq-mq`.
- If vectorstore already exists, it loads it instead of re-creating it.

---


> "Supporting mental health professionals with the power of AI."

