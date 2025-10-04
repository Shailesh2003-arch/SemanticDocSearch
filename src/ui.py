import gradio as gr
from retrieval import ask_question

def chat_with_docs(message, history):
    response = ask_question(message)
    return response

demo = gr.ChatInterface(fn=chat_with_docs, title="📚 Chat with My Docs 🤖")
demo.launch()
