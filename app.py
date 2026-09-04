import os
import uvicorn
import gradio as gr
from api import app as fastapi_app

# ZeroGPU setup: Hugging Face ZeroGPU requires at least one @spaces.GPU function during startup
try:
    import spaces
    @spaces.GPU
    def zero_gpu_warmup():
        """Dummy GPU function to satisfy Hugging Face ZeroGPU startup check."""
        return True
except Exception:
    pass

with gr.Blocks(title="NutriScale AI Recommendation Engine") as demo:
    gr.Markdown("# 🥗 NutriScale AI Recommendation Engine")
    gr.Markdown("FastAPI ML + Gemini LLM Microservice active on Hugging Face Spaces.")
    gr.Markdown("API Endpoints active at `/api/recommend`, `/api/sync`, `/metrics`.")

# Mount Gradio UI cleanly onto the main FastAPI application at /ui
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
