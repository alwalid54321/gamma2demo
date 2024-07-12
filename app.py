import gradio as gr
from gradio_client import Client

# Initialize the client
client = Client("huggingface-projects/gemma-2-9b-it")

# Define a function that uses the client to get predictions
def get_response(message):
    result = client.predict(
        message=message,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        api_name="/chat"
    )
    return result

# Create a Gradio interface
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
    outputs="text",
    title="Hugging Face Chatbot",
    description="A chatbot demo using the Hugging Face Gemma-2-9b-it model."
)

# Launch the interface
iface.launch()
