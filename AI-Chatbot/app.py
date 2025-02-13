import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define chatbot function
def chat_with_ai(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return bot_response

# Create Gradio UI
chatbot_ui = gr.Interface(
    fn=chat_with_ai,
    inputs=gr.Textbox(placeholder="Type a message..."),
    outputs="text",
    title="AI Chatbot",
    description="A simple AI chatbot using Hugging Face Transformers."
)

chatbot_ui.launch()
