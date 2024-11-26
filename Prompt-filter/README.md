
# Prompt Filter API

This project implements a Flask-based API that filters input prompts using a combination of keyword matching and embeddings-based similarity. It checks if a prompt contains sensitive keywords or exceeds a threshold similarity with the provided keywords.

## Requirements

Before running this application, make sure you have the following dependencies installed:

- Python 3.6+ 
- Flask
- Sentence-Transformers
- PyTorch

You can install the required dependencies by running:

```bash
pip install flask sentence-transformers torch
```

## Setup

1. **Download the necessary model**:
   The app uses the `all-MiniLM-L6-v2` model from the `sentence-transformers` library to compute embeddings. The model will automatically be downloaded when you run the app for the first time.

2. **Create `filter_keywords.json` file**:
   This file contains a list of sensitive keywords that the app will check against. Here is an example of the JSON format:

   ```json
   {
     "keywords": [
       "sensitive term 1",
       "sensitive term 2",
       "confidential phrase"
     ]
   }
   ```

   The app will load these keywords to check if they exist in the input prompt.

## Running the Application

To run the Flask app locally, execute the following command in the project directory:

```bash
python your_script_name.py
```

This will start a local server at `http://127.0.0.1:5000`.

## API Endpoint

The app exposes the following API endpoint:

### POST `/filter_prompt`

This endpoint checks if a given prompt is safe. It performs two checks:
1. **Embedding similarity check**: Compares the prompt embedding with the embeddings of the keywords.
2. **Keyword matching check**: Checks if any of the sensitive keywords appear in the prompt text.

#### Request Body

```json
{
  "prompt": "Your input prompt here"
}
```

- `prompt`: The input text to be evaluated.

#### Response

The response will be a JSON object indicating whether the prompt is safe or not.

```json
{
  "prompt": "Your input prompt here",
  "safe": true,
  "message": "Prompt is safe"
}
```

If the prompt contains sensitive keywords or exceeds the similarity threshold, the response will be:

```json
{
  "prompt": "Your input prompt here",
  "safe": false,
  "message": "Prompt is not safe"
}
```

If the prompt is missing or invalid, you will get an error response:

```json
{
  "error": "Prompt is required"
}
```

In case of other errors:

```json
{
  "error": "Error message here"
}
```

## Testing the API

You can test the API using tools like **Postman** or **cURL**.

### Example with cURL:

```bash
curl -X POST http://127.0.0.1:5000/filter_prompt \
-H "Content-Type: application/json" \
-d '{"prompt": "example sensitive term"}'
```

### Example with Postman:

1. Set the HTTP method to `POST`.
2. Enter the URL `http://127.0.0.1:5000/filter_prompt`.
3. In the body section, choose **raw** and **JSON** format.
4. Add your input data:
   ```json
   {
     "prompt": "example sensitive term"
   }
   ```
5. Send the request and check the response.

## Customizing the Threshold

The similarity threshold used to compare embeddings is set to `0.7` by default. You can modify it in the function `is_prompt_safe_with_embeddings`:

```python
def is_prompt_safe_with_embeddings(prompt, keywords, threshold=0.7):
```

Change the `threshold` value to a higher or lower number to adjust the sensitivity of the check.