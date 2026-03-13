import base64
import io
import os
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

app = FastAPI(title="TADA Text-to-Speech API", version="1.0.0")

# Global variables for models
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = None
model = None


@app.on_event("startup")
async def startup_event():
    global encoder, model
    print(f"Loading models on {device}...")

    # Patch AutoTokenizer to support local Llama tokenizer paths
    from transformers import AutoTokenizer

    original_from_pretrained = AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "meta-llama/Llama-3.2-1B":
            local_path = os.environ.get("LLAMA_LOCAL_PATH")
            if local_path and os.path.exists(local_path):
                pretrained_model_name_or_path = local_path
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = patched_from_pretrained

    # Check if a different model path is provided via environment variables
    encoder_path = os.environ.get("TADA_ENCODER_PATH", "HumeAI/tada-codec")
    model_path = os.environ.get("TADA_MODEL_PATH", "HumeAI/tada-1b")

    try:
        print(f"Loading encoder from {encoder_path} in bfloat16")
        encoder = Encoder.from_pretrained(encoder_path, subfolder="encoder").to(torch.bfloat16).to(device)

        print(f"Loading model from {model_path} in bfloat16")
        model = TadaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        print("Models loaded successfully.")
    except Exception as e:
        import traceback

        print(f"Error loading models: {e}")
        traceback.print_exc()
        # Allow the app to start even if models fail to load so we can return 500s
        # instead of failing completely, useful for debugging in Swarm.


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    prompt_text: Optional[str] = Field(None, description="Text of the prompt audio for voice cloning/continuation")
    prompt_audio_base64: Optional[str] = Field(None, description="Base64 encoded audio file for voice cloning")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'ja', 'es') if non-English")
    num_extra_steps: int = Field(0, description="Number of extra steps for speech continuation")
    speed_up_factor: Optional[float] = Field(None, description="Speed up factor for the generated audio")


def load_audio_from_base64(base64_str: str) -> tuple[torch.Tensor, int]:
    """Decodes a base64 audio string to a PyTorch tensor."""
    try:
        # Check for data URI prefix and remove it if present
        if base64_str.startswith("data:audio/"):
            base64_str = base64_str.split(",")[1]

        audio_data = base64.b64decode(base64_str)
        audio_io = io.BytesIO(audio_data)
        audio_tensor, sample_rate = torchaudio.load(audio_io)
        return audio_tensor, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio: {e}")


@app.get("/health")
def health_check():
    if model is None or encoder is None:
        return {"status": "unhealthy", "message": "Models not loaded", "device": device}
    return {"status": "healthy", "device": device}


@app.post("/generate")
async def generate_audio(request: GenerateRequest):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Models are not loaded or currently initializing.")

    try:
        # Generate the audio
        with torch.no_grad():
            # Handle conditioning / voice cloning
            prompt = None
            if request.prompt_audio_base64:
                if not request.prompt_text:
                    raise HTTPException(
                        status_code=400, detail="prompt_text is required when prompt_audio_base64 is provided."
                    )

                try:
                    audio_tensor, sample_rate = load_audio_from_base64(request.prompt_audio_base64)
                    # Ensure the audio tensor on GPU matches the model's dtype (bfloat16)
                    audio_tensor = audio_tensor.to(device).to(torch.bfloat16)

                    # Default to None, but will use language if we had initialized an encoder for it.
                    # Note: Currently the encoder object is fixed to the loaded language.
                    # For a full multilingual API, we might need to load multiple encoders or reload.

                    prompt = encoder(audio_tensor, text=[request.prompt_text], sample_rate=sample_rate)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing prompt audio: {e}")
            else:
                # If no prompt audio is provided, we need a default prompt to initialize generation.
                # TADA typically requires some acoustic features to kick off.
                # We'll use a silent prompt or check if the model supports unconditional generation.
                # Here we load a default sample if no prompt is provided.
                default_audio_path = os.path.join(os.path.dirname(__file__), "tada", "samples", "ljspeech.wav")
                if os.path.exists(default_audio_path):
                    audio_tensor, sample_rate = torchaudio.load(default_audio_path)
                    # Ensure the audio tensor on GPU matches the model's dtype (bfloat16)
                    audio_tensor = audio_tensor.to(device).to(torch.bfloat16)
                    default_text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
                    prompt = encoder(audio_tensor, text=[default_text], sample_rate=sample_rate)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="prompt_audio_base64 and prompt_text are required (no default sample found).",
                    )

            output = model.generate(prompt=prompt, text=request.text, num_extra_steps=request.num_extra_steps)

        generated_audio = output.audio[0]  # Take the first batch item

        if generated_audio is None:
            raise HTTPException(status_code=500, detail="Model failed to generate audio.")

        # Save to buffer
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            generated_audio.to(torch.float32).cpu().unsqueeze(0),  # torchaudio expects [channels, time] in float32
            24000,  # TADA default sample rate
            format="wav",
        )
        buffer.seek(0)

        return StreamingResponse(
            buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=generated.wav"}
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


# Add a simple main block to support running with python api.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
