import gradio as gr
from api import app as fastapi_app

# Gradio Interface dashboard for Hugging Face Space UI
demo = gr.Interface(
    fn=lambda msg: "🥗 NutriScale AI Recommendation Engine is RUNNING! POST /api/recommend endpoint is ready.",
    inputs=gr.Textbox(label="System Check", value="Ping"),
    outputs=gr.Textbox(label="System Status"),
    title="🥗 NutriScale AI Recommendation Engine",
    description="FastAPI ML + Gemini LLM Microservice hosted on Hugging Face Spaces."
)

# Mount Gradio at /ui so FastAPI API endpoints (/api/recommend, /health) remain clean at root /
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    demo.launch()
