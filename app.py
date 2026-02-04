import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from TTS.utils.synthesizer import Synthesizer
from dotenv import load_dotenv

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)
CORS(app)

load_dotenv()

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# Load Sinhala TTS model (GPU)
# ----------------------------
synthesizer = Synthesizer(
    tts_checkpoint="./tts-si-female-vits-v2/model.pth",
    tts_config_path="./tts-si-female-vits-v2/config.json",
    use_cuda=True
)

# ----------------------------
# Helper function to generate TTS
# ----------------------------
def synthesize_sinhala(text: str, output_path: str):
    """
    Generate speech using low-level Synthesizer API and save to file.
    """
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, output_path)
    return output_path

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        logger.info("=" * 50)
        data = request.json
        user_message = data.get("message")
        history = data.get("history", [])

        if not user_message:
            return jsonify({"error": "A message is required"}), 400

        # Build conversation messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in Sinhala language. Always respond in Sinhala (සිංහල) to the user's questions regardless of the language they use."
            }
        ]

        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        logger.info(f"Total messages in conversation: {len(messages)}")
        logger.info("Making API call to OpenRouter/OpenAI...")

        # OpenRouter/OpenAI API key
        API_KEY = OPENROUTER_API_KEY

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

        # Make API call
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages
        )

        assistant_message = response.choices[0].message.content
        logger.info(f"Assistant response: {assistant_message[:100]}...")

        # Generate unique WAV file
        audio_id = str(uuid.uuid4())
        audio_path = f"static/responses/audio_{audio_id}.wav"
        os.makedirs("static/responses", exist_ok=True)

        # Generate TTS
        synthesize_sinhala(assistant_message, audio_path)

        return jsonify({
            "response": assistant_message,
            "audio_url": f"http://localhost:5000/{audio_path}",
            "success": True
        })

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"An Error has occured: {str(e)}",
            "success": False
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
