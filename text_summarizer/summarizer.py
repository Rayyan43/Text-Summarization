import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer


# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define the summarization function
def summarize_text(text, max_length=130, min_length=30):
    if not text.strip():
        return "Error: Input text is empty!"

    try:
        # Tokenize the input text
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)

        # Generate summary using the model
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}"


# Define evaluation function using ROUGE score
def evaluate_summary(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores


# Main function to test the summarizer
if __name__ == "__main__":
    # Sample text input
    text = """
    The COVID-19 pandemic has caused a significant disruption across the world, affecting the global economy, 
    healthcare systems, and daily lives. Governments have implemented various measures to contain the spread, 
    including lockdowns and vaccination drives. Despite these efforts, the pandemic continues to pose challenges 
    in terms of new variants and healthcare capacity.
    """

    # Generate summary
    summary = summarize_text(text)
    print("Original Text:\n", text)
    print("\nGenerated Summary:\n", summary)

    # Evaluate the generated summary
    reference = "The pandemic disrupted the global economy and healthcare systems, with lockdowns and vaccination drives."
    scores = evaluate_summary(reference, summary)
    print("\nEvaluation Scores:\n", scores)
