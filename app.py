import gradio as gr
from api import app as fastapi_app

def check_system_status(input_text):
    return "🥗 NutriScale AI Recommendation Engine is RUNNING! POST /api/recommend endpoint is active."

with gr.Blocks(title="NutriScale AI Recommendation Engine") as demo:
    gr.Markdown("# 🥗 NutriScale AI Recommendation Engine")
    gr.Markdown("FastAPI ML + Gemini LLM Microservice hosted on Hugging Face Spaces.")
    input_box = gr.Textbox(label="System Check", value="Ping")
    output_box = gr.Textbox(label="System Status", value="🥗 NutriScale AI Recommendation Engine is RUNNING! POST /api/recommend endpoint is active.")
    input_box.submit(fn=check_system_status, inputs=input_box, outputs=output_box)

# Mount Gradio at /ui so FastAPI API endpoints (/api/recommend, /health) remain clean at root /
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    demo.launch()
