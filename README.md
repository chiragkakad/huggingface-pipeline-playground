# 🤗 Hugging Face Pipeline Playground

An interactive **Gradio** web app to test and compare Hugging Face pipelines for different NLP and CV tasks — including **sentiment analysis**, **summarization**, **translation**, **NER**, **text generation**, **image classification**, and **object detection** — with live model switching.


---

## ✨ Features
- **Task Dropdown** — Choose from multiple NLP & CV tasks.
- **Model Dropdown** — Automatically updates with task-relevant models.
- **Markdown Output** — Results displayed as clean, copyable tables.
- **Fast Model Loading** — Caches pipelines so switching is instant after first load.
- **LAN-Accessible** — Share your local server across devices.

---

## 📋 Supported Tasks & Models

| Task                       | Input Type | Models |
|----------------------------|------------|--------|
| Sentiment Analysis         | Text       | `distilbert-base-uncased-finetuned-sst-2-english`, `cardiffnlp/twitter-roberta-base-sentiment` |
| Text Summarization         | Text       | `facebook/bart-large-cnn`, `t5-small` |
| Text Translation (EN→FR)   | Text       | `Helsinki-NLP/opus-mt-en-fr` |
| Text Generation            | Text       | `gpt2`, `distilgpt2` |
| Named Entity Recognition   | Text       | `dslim/bert-base-NER`, `dbmdz/bert-large-cased-finetuned-conll03-english` |
| Image Classification       | Image      | `google/vit-base-patch16-224`, `microsoft/resnet-50` |
| Object Detection           | Image      | `facebook/detr-resnet-50`, `hustvl/yolos-tiny` |

---

## 🚀 Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/chiragkakad/huggingface-pipeline-playground.git
   cd huggingface-pipeline-playground
```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---


## ▶️ Run the app

```bash
python app.py
```

By default, the app will be available at:

```
http://127.0.0.1:7860
```

To make it accessible from other devices on your network:

```bash
python app.py --server-name 0.0.0.0
```

---

## 📸 Example Usage

**Sentiment Analysis Example:**

```
Task: Sentiment Analysis
Model: distilbert-base-uncased-finetuned-sst-2-english
Text: "I love using open-source models!"

Output:
| label     | score   |
|-----------|---------|
| POSITIVE  | 0.99987 |
```

---

## 📂 Project Structure

```
huggingface-pipeline-playground/
├── app.py              # Main Gradio app
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
```

---

## 🙌 Acknowledgments

* [Hugging Face](https://huggingface.co/) for pipelines and models.
* [Gradio](https://gradio.app/) for the web UI framework.

```

---