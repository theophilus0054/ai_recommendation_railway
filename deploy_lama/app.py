import os
import uvicorn
import spaces
import gradio as gr
from api import app as fastapi_app

# Top-level @spaces.GPU decoration required by HF ZeroGPU static inspector
@spaces.GPU
def zero_gpu_init():
    """Top-level GPU initialization function for Hugging Face ZeroGPU runtime."""
    return True

# Trigger on startup
zero_gpu_init()

with gr.Blocks(title="NutriScale AI Recommendation Engine") as demo:
    gr.Markdown("# 🥗 NutriScale AI Recommendation Engine")
    gr.Markdown("FastAPI ML + Gemini LLM Microservice active on Hugging Face Spaces.")
    gr.Markdown("API Endpoints active at `/api/recommend`, `/api/sync`, `/metrics`.")

# Mount Gradio UI onto FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
