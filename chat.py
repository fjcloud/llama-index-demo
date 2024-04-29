from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
import gradio as gr
import sys
import time
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir="storage")

Settings.llm = Groq(model="llama3-70b-8192", temperature=0, context_window=8192)

index = load_index_from_storage(storage_context)

with gr.Blocks() as demo:
    chat_engine = index.as_chat_engine(
    similarity_top_k=4,
    chat_mode="context",
    system_prompt=(
        "You are a chatbot tech expert on AWS and ROSA, you will carefully respond to each question of customer with some bullets points and docs reference. If you are not sure tell them to refer to fjacquin@redhat.com"
    ),
    )

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="⏎ for sending",
            placeholder="Ask me something",)
    clear = gr.Button("Delete")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_engine.chat(user_message)
        history[-1][1] = ""
        for character in bot_message.response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=True).then(
        bot, chatbot, chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=True)

demo.queue().launch(share=True, server_name="0.0.0.0")
