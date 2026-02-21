import json
import os

from flask import Flask, render_template, request, jsonify

import deploy

app = Flask(__name__)


REQUIRED_MODEL_FILES = [
    "bernoullinb.sav",
    "countvectorizer.sav",
    "labelencoder_1.sav",
    "labelencoder_2.sav",
    "labelencoder_3.sav",
    "columntransformer1.sav",
    "columntransformer2.sav",
    "columntransformer3.sav",
]


def _models_ready():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return all(os.path.exists(os.path.join(models_dir, fname)) for fname in REQUIRED_MODEL_FILES)


def _load_metrics():
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


@app.route("/")
def homepage():
    return render_template("template.htm")


@app.route("/health", methods=["GET"])
def health():
    metrics = _load_metrics()
    return jsonify(
        {
            "status": "ok",
            "models_ready": _models_ready(),
            "metrics": metrics,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()

    review_text = (payload.get("review_text") or "").strip()
    rating = str(payload.get("rating", "")).strip()
    verified_purchase = str(payload.get("verified_purchase", "")).strip().upper()
    product_category = str(payload.get("product_category", "")).strip()

    if not review_text:
        return jsonify({"ok": False, "errors": ["Review text is required."]}), 400

    try:
        result = deploy.predict_review(review_text, rating, verified_purchase, product_category)
    except LookupError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "errors": [
                        "NLTK data is missing. Run: python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')\"",
                        str(exc),
                    ],
                }
            ),
            500,
        )
    except Exception as exc:
        return jsonify({"ok": False, "errors": [str(exc)]}), 500

    if not result["ok"]:
        return jsonify(result), 400

    metrics = _load_metrics()
    result["metrics"] = metrics
    return jsonify(result)


def config_ngrok():
    from pyngrok import ngrok

    port = int(os.environ.get("PORT", "5000"))
    url = ngrok.connect(port)
    print(" * Public URL:", url)


if __name__ == "__main__":
    use_ngrok = os.environ.get("USE_NGROK", "0") == "1"
    if use_ngrok:
        config_ngrok()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
