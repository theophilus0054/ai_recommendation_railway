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

# Mount fastapi_app (routes & lifespan) into demo's underlying FastAPI app
demo.app.mount("/", fastapi_app)

# Launch via Gradio's native Hugging Face Space launcher
demo.launch()
