import gradio as gr
from transformers import pipeline
import pandas as pd

# ----- Task & Model config -----
TASKS = {
    "Sentiment Analysis": {
        "task_id": "sentiment-analysis",
        "input_type": "text",
        "models": [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "cardiffnlp/twitter-roberta-base-sentiment"
        ],
        "example": "I love using open-source models!"
    },
    "Text Summarization": {
        "task_id": "summarization",
        "input_type": "text",
        "models": [
            "facebook/bart-large-cnn",
            "t5-small"
        ],
        "example": (
            "Hugging Face hosts models, datasets, and spaces that enable practitioners "
            "to build state-of-the-art machine learning applications quickly."
        )
    },
    "Text Translation": {
        "task_id": "translation_en_to_fr",
        "input_type": "text",
        "models": [
            "Helsinki-NLP/opus-mt-en-fr"
        ],
        "example": "Good morning! How are you today?"
    },
    "Text Generation": {
        "task_id": "text-generation",
        "input_type": "text",
        "models": [
            "gpt2",
            "distilgpt2"
        ],
        "example": "Once upon a time"
    },
    "Named Entity Recognition": {
        "task_id": "ner",
        "input_type": "text",
        "models": [
            "dslim/bert-base-NER",
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        ],
        "example": "Barack Obama was born in Hawaii."
    },
    "Image Classification": {
        "task_id": "image-classification",
        "input_type": "image",
        "models": [
            "google/vit-base-patch16-224",
            "microsoft/resnet-50"
        ]
    },
    "Object Detection": {
        "task_id": "object-detection",
        "input_type": "image",
        "models": [
            "facebook/detr-resnet-50",
            "hustvl/yolos-tiny"
        ]
    }
}

# Cache pipelines so they don't reload each time
PIPELINE_CACHE = {}

def get_pipeline(task_id: str, model_name: str):
    key = (task_id, model_name)
    if key not in PIPELINE_CACHE:
        PIPELINE_CACHE[key] = pipeline(task_id, model=model_name)
    return PIPELINE_CACHE[key]

# ----- Formatting helpers -----
def to_markdown_table(obj):
    """
    Convert common pipeline outputs to a Markdown table string.
    - list[dict] -> table of keys/values
    - dict -> single-row table
    - list[str] or str -> code block / plain string
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return "_No results_"
        if isinstance(obj[0], dict):
            df = pd.DataFrame(obj)
            return df.to_markdown(index=False)
        else:
            # list of strings/numbers
            return "```\n" + "\n".join(map(str, obj)) + "\n```"
    elif isinstance(obj, dict):
        df = pd.DataFrame([obj])
        return df.to_markdown(index=False)
    else:
        return str(obj)

# ----- Inference -----
def run_pipeline(task_name, model_name, text_input, image_input):
    task = TASKS[task_name]
    task_id = task["task_id"]
    pipe = get_pipeline(task_id, model_name)

    if task["input_type"] == "text":
        if not text_input or not text_input.strip():
            return "_Please enter some text input._"
        result = pipe(text_input)
        return to_markdown_table(result)

    elif task["input_type"] == "image":
        if image_input is None or image_input == "":
            return "_Please upload an image._"
        result = pipe(image_input)
        # For object detection, normalize the bounding box dict for a nicer table
        if task_id == "object-detection" and isinstance(result, list):
            norm = []
            for r in result:
                row = {k: v for k, v in r.items() if k != "box"}
                # box -> a compact string
                if "box" in r and isinstance(r["box"], dict):
                    row["box"] = f"x={r['box'].get('xmin')}, y={r['box'].get('ymin')}, w={r['box'].get('xmax')}, h={r['box'].get('ymax')}"
                norm.append(row)
            return to_markdown_table(norm)
        return to_markdown_table(result)

    return "_Unsupported task/input combination._"

# ----- UI dynamic updates -----
def on_task_change(task_name):
    """Update model dropdown choices/value and toggle input visibility."""
    task = TASKS[task_name]
    models = task["models"]
    default_model = models[0] if models else ""
    is_text = (task["input_type"] == "text")
    example = task.get("example", "")

    return (
        gr.update(choices=models, value=default_model, label="Model"),  # model dropdown
        gr.update(visible=is_text, value=example if is_text else ""),   # text input
        gr.update(visible=not is_text)                                  # image input
    )

# ----- Build UI -----
with gr.Blocks() as demo:
    gr.Markdown("## 🤗 Hugging Face Pipeline Playground")

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=list(TASKS.keys()),
            value="Sentiment Analysis",
            label="Task"
        )
        model_dropdown = gr.Dropdown(
            choices=TASKS["Sentiment Analysis"]["models"],
            value=TASKS["Sentiment Analysis"]["models"][0],
            label="Model"
        )

    text_input = gr.Textbox(
        label="Text Input",
        value=TASKS["Sentiment Analysis"]["example"],
        lines=4,
        visible=True,
        placeholder="Type or paste your text..."
    )
    image_input = gr.Image(
        type="filepath",
        label="Image Input",
        visible=False
    )

    run_btn = gr.Button("Run Inference")
    output_md = gr.Markdown()

    # wire up dynamic UI updates
    task_dropdown.change(
        fn=on_task_change,
        inputs=task_dropdown,
        outputs=[model_dropdown, text_input, image_input]
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[task_dropdown, model_dropdown, text_input, image_input],
        outputs=output_md
    )

if __name__ == "__main__":
    # LAN-ready
    demo.launch(server_name="0.0.0.0", server_port=7860)
