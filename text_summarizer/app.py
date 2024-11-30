from flask import Flask, request, jsonify, render_template
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from summarizer import summarize_text, evaluate_summary

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Summarization route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["input_text"]
        max_length = int(request.form.get("max_length", 130))
        min_length = int(request.form.get("min_length", 30))

        # Summarize text
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return render_template("index.html", summary=summary)
    return render_template("index.html", summary=None)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)

    #http://127.0.0.1:5000
