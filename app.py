"""Step 2: Gradio chat UI. Run: python app.py"""
import gradio as gr
import config
from rag import answer, REFUSAL


def chat(message, history):
    text, sources = answer(message)
    if text.strip().startswith(REFUSAL[:30]) or not sources:
        return f"{text}\n\n_{config.DISCLAIMER}_"
    return f"{text}\n\n**Sources:** {', '.join(sources)}\n\n_{config.DISCLAIMER}_"


demo = gr.ChatInterface(
    chat,
    title="Healthcare FAQ Chatbot (RAG)",
    description="Answers only from a small set of public health documents. Not medical advice.",
    examples=["What are the symptoms of type 2 diabetes?", "How can I lower my blood pressure?"],
)

if __name__ == "__main__":
    demo.launch(share=True)
