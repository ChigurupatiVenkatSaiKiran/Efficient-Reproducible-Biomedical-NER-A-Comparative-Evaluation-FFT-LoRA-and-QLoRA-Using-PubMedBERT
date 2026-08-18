"""
PubMedBERT Biomedical NER — Chemical & Disease Finder
Standalone HuggingFace Spaces app.
Loads Full / LoRA / QLoRA fine-tuned models and highlights
Chemical / Disease entities in pasted text or uploaded PDFs.

Same inference logic as the notebook demo (cell 22-23), adapted to
load models from the HuggingFace Hub instead of Google Drive.
"""

import os
import numpy as np
import torch
import gradio as gr

try:
    import spaces  # only available when running on a HuggingFace ZeroGPU Space
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ─── CONFIG ──────────────────────────────────────────────────────────────
# Base model used for fine-tuning
MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# ─── EDIT THIS ───────────────────────────────────────────────────────────
# Your HuggingFace Hub repo(s) where you uploaded the fine-tuned models.
# You can use ONE repo with subfolders, or separate repos per method.
# Example: if you uploaded to "your-username/pubmedbert-bc5cdr-ner" with
# subfolders "Full", "LoRA", "QLoRA" inside it, this works as-is.
HF_REPO = os.environ.get("HF_MODEL_REPO", "Venkatsaikiran/pubmedbert-bc5cdr-ner")

MODEL_SUBFOLDERS = {
    "Full":  "Full",
    "LoRA":  "LoRA",
    "QLoRA": "QLoRA",
}

# BC5CDR label scheme (matches the notebook)
LABEL_LIST = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
id2label = {i: l for i, l in enumerate(LABEL_LIST)}
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

CUDA_OK = torch.cuda.is_available()

ENTITY_COLORS = {
    "Chemical": {"bg": "#cce5ff", "border": "#0066cc"},
    "Disease":  {"bg": "#ffd6d6", "border": "#cc0000"},
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

_model_cache = {}


def _load_inference_model(method):
    if method in _model_cache:
        return _model_cache[method]

    subfolder = MODEL_SUBFOLDERS[method]
    print(f"Loading {method} from {HF_REPO}/{subfolder} ...")

    if method == "Full":
        model = AutoModelForTokenClassification.from_pretrained(
            HF_REPO, subfolder=subfolder,
            num_labels=NUM_LABELS, id2label=id2label, label2id=label2id,
        )

    elif method == "LoRA":
        base = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, HF_REPO, subfolder=subfolder)

    elif method == "QLoRA":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["classifier"],
        )
        base = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id,
            quantization_config=bnb_cfg if CUDA_OK else None,
            ignore_mismatched_sizes=True,
            device_map="auto" if CUDA_OK else None,
        )
        model = PeftModel.from_pretrained(base, HF_REPO, subfolder=subfolder)

    else:
        raise ValueError(f"Unknown method: {method}")

    model.eval()
    if method != "QLoRA":
        model = model.to("cuda" if CUDA_OK else "cpu")

    _model_cache[method] = model
    print(f"{method} loaded and cached.")
    return model


def _run_ner(text, model, max_len=512):
    device = next(model.parameters()).device
    words = text.split()
    enc = tokenizer(
        words, is_split_into_words=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    pred_probs = probs[0].max(dim=-1).values.tolist()
    word_ids = enc.word_ids(batch_index=0)

    word_preds = {}
    for tok_i, word_i in enumerate(word_ids):
        if word_i is None or word_i in word_preds:
            continue
        word_preds[word_i] = (id2label[pred_ids[tok_i]], pred_probs[tok_i])

    entities, cur_words, cur_type, cur_conf = [], [], None, []
    for w_i, word in enumerate(words):
        tag, conf = word_preds.get(w_i, ("O", 1.0))
        etype = tag[2:] if tag != "O" else None
        if tag.startswith("B-"):
            if cur_words:
                entities.append({"text": " ".join(cur_words), "type": cur_type,
                                  "confidence": round(float(np.mean(cur_conf)), 3)})
            cur_words, cur_type, cur_conf = [word], etype, [conf]
        elif tag.startswith("I-") and cur_type == etype:
            cur_words.append(word); cur_conf.append(conf)
        else:
            if cur_words:
                entities.append({"text": " ".join(cur_words), "type": cur_type,
                                  "confidence": round(float(np.mean(cur_conf)), 3)})
            cur_words, cur_type, cur_conf = [], None, []
    if cur_words:
        entities.append({"text": " ".join(cur_words), "type": cur_type,
                          "confidence": round(float(np.mean(cur_conf)), 3)})
    return entities


def _html_highlight(text, entities):
    if not entities:
        return f'<p style="font-size:15px;line-height:1.8">{text}</p>'
    found = []
    for ent in entities:
        idx = text.lower().find(ent["text"].lower())
        if idx != -1:
            found.append((idx, idx + len(ent["text"]), ent))
    found.sort(key=lambda x: x[0])
    html, prev = [], 0
    for start, end, ent in found:
        if start < prev:
            continue
        html.append(text[prev:start])
        col = ENTITY_COLORS.get(ent["type"], {"bg": "#eee", "border": "#888"})
        conf = int(ent["confidence"] * 100)
        html.append(
            '<mark title="' + ent["type"] + " - " + str(conf) + '% confidence" '
            'style="background:' + col["bg"] + ';border:1.5px solid ' + col["border"] + ";"
            'border-radius:4px;padding:2px 5px;font-weight:600;cursor:help;">'
            + text[start:end] + "</mark>"
        )
        prev = end
    html.append(text[prev:])
    body = "".join(html)
    return ('<div style="font-size:15px;line-height:2.2;font-family:Georgia,serif;'
            'padding:14px;border-radius:8px;background:#fafafa;">' + body + "</div>")


def _predict_core(text, method):
    try:
        model = _load_inference_model(method)
    except Exception as e:
        return "<p style='color:red'>ERROR: " + str(e) + "</p>", "", ""
    entities = _run_ner(text, model)
    return _finish_prediction(text, entities)


if HAS_ZEROGPU:
    _predict_core = spaces.GPU(_predict_core)


def _finish_prediction(text, entities):
    chemicals = [e for e in entities if e["type"] == "Chemical"]
    diseases = [e for e in entities if e["type"] == "Disease"]

    stats = (f"Chemicals found : {len(chemicals)}\n"
             f"Diseases found  : {len(diseases)}\n"
             f"Total entities  : {len(entities)}")

    lines = []
    if chemicals:
        lines.append("CHEMICALS:")
        lines += [f"   - {e['text']} ({int(e['confidence']*100)}% confidence)" for e in chemicals]
    if diseases:
        lines.append("DISEASES:")
        lines += [f"   - {e['text']} ({int(e['confidence']*100)}% confidence)" for e in diseases]
    entity_list = "\n".join(lines) if lines else "No entities found."
    return _html_highlight(text, entities), stats, entity_list


def predict_and_format(text, method):
    text = (text or "").strip()
    if not text:
        return "<p style='color:gray'>Enter some biomedical text above.</p>", "", ""
    return _predict_core(text, method)


def extract_text_from_pdf(pdf_file):
    try:
        import fitz
        doc = fitz.open(pdf_file)
        pages = [page.get_text() for page in doc][:5]  # first 5 pages
        return " ".join(" ".join(p.split()) for p in pages)
    except Exception as e:
        return "Could not read PDF: " + str(e)


def analyze_pdf(pdf_file, method):
    if pdf_file is None:
        return "<p style='color:gray'>Upload a PDF above.</p>", "", ""
    text = extract_text_from_pdf(pdf_file.name)[:3000]
    return predict_and_format(text, method)


EXAMPLES = [
    "The patient was treated with metformin for type 2 diabetes and developed lactic acidosis.",
    "Long-term lithium therapy is associated with an increased risk of hypothyroidism and renal impairment.",
    "Aspirin overdose can cause tinnitus, metabolic acidosis, and in severe cases, pulmonary edema.",
    "Chronic use of nonsteroidal anti-inflammatory drugs may lead to gastric ulcers and hypertension.",
    "Methotrexate is commonly prescribed for rheumatoid arthritis but can cause hepatotoxicity.",
]

CUSTOM_CSS = """
#title-heading h1, #title-heading * {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px;
}
#header-banner {
    text-align: center;
    padding: 28px 16px 20px 16px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0b3d91 0%, #1e5fbf 45%, #a3122b 100%);
    margin-bottom: 18px;
}
#header-banner h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    margin-bottom: 6px !important;
}
#header-banner p {
    color: #e8eefc !important;
    font-size: 1rem !important;
    margin: 0 !important;
}
.legend-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin: 0 4px;
}
#footer-badges {
    text-align: center;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #333;
    font-size: 0.85rem;
    opacity: 0.85;
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="rose",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

with gr.Blocks(title="PubMedBERT Biomedical NER", css=CUSTOM_CSS, theme=THEME) as demo:
    gr.HTML(
        """
        <div id="header-banner">
            <h1>🧬 PubMedBERT Biomedical NER</h1>
            <p>Chemical &amp; Disease entity recognition on BC5CDR ·
               <span class="legend-pill" style="background:#cce5ff;color:#0047ab;">🟦 Chemical</span>
               <span class="legend-pill" style="background:#ffd6d6;color:#a3122b;">🟥 Disease</span>
            </p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Paste Text"):
            method_dd1 = gr.Dropdown(["Full", "LoRA", "QLoRA"], value="LoRA", label="Model")
            text_in = gr.Textbox(lines=4, label="Biomedical text", placeholder="Paste a sentence or paragraph...")
            btn1 = gr.Button("Analyze", variant="primary")
            html_out1 = gr.HTML(label="Highlighted text")
            with gr.Row():
                stats_out1 = gr.Textbox(label="Summary", lines=3)
                list_out1 = gr.Textbox(label="Entities found", lines=6)
            gr.Examples(examples=[[e, "LoRA"] for e in EXAMPLES], inputs=[text_in, method_dd1])
            btn1.click(predict_and_format, inputs=[text_in, method_dd1],
                       outputs=[html_out1, stats_out1, list_out1])

        with gr.Tab("Upload PDF"):
            method_dd2 = gr.Dropdown(["Full", "LoRA", "QLoRA"], value="LoRA", label="Model")
            pdf_in = gr.File(label="Upload a medical PDF", file_types=[".pdf"])
            btn2 = gr.Button("Analyze PDF", variant="primary")
            html_out2 = gr.HTML(label="Highlighted text (first 5 pages, 3000 chars)")
            with gr.Row():
                stats_out2 = gr.Textbox(label="Summary", lines=3)
                list_out2 = gr.Textbox(label="Entities found", lines=6)
            btn2.click(analyze_pdf, inputs=[pdf_in, method_dd2],
                       outputs=[html_out2, stats_out2, list_out2])

        with gr.Tab("About"):
            gr.Markdown(
                """
                ### About this project
                - **Base model:** microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
                - **Dataset:** BC5CDR (Chemical & Disease NER)
                - **Methods compared:** Full fine-tuning, LoRA, QLoRA
                - **Evaluation:** seqeval, validation + held-out test set

                Switch the **Model** dropdown to compare how Full / LoRA / QLoRA
                perform on the same text.
                """
            )

if __name__ == "__main__":
    demo.launch()
