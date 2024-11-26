from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)

# Load model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load keywords
def load_filter_keywords():
    with open("filter_keywords.json", "r") as file:
        return json.load(file)["keywords"]

# Function to check prompt safety using embeddings
def is_prompt_safe_with_embeddings(prompt, keywords, threshold=0.7):
    # Compute embeddings
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(prompt_embedding, keyword_embeddings)

    # Check if any similarity exceeds the threshold
    for sim in similarities[0]:
        if sim > threshold:
            return False
    return True

# Function to check if prompt contains any keyword
def contains_keywords(prompt, keywords):
    for keyword in keywords:
        if keyword.lower() in prompt.lower():
            return True
    return False

@app.route("/filter_prompt", methods=["POST"])
def filter_prompt():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Load keywords
        keywords = load_filter_keywords()

        # Check prompt safety
        safe = is_prompt_safe_with_embeddings(prompt, keywords) and not contains_keywords(prompt, keywords)

        return jsonify({
            "prompt": prompt,
            "safe": safe,
            "message": "Prompt is safe" if safe else "Prompt is not safe"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)