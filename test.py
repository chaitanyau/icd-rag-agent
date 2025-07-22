import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
import time

# === Vector DB ===
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectordb = Chroma(persist_directory="embeddings", embedding_function=embedding)

# === Local LLM ===
llm = OllamaLLM(model="phi3:medium")

# === ICD-11 Retrieval ===
def retrieve_icd_context(query, top_k=4):
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)  # ✅ new recommended API
    if not docs:
        return "", []
    combined_context = "\n\n".join([d.page_content for d in docs])
    return combined_context.strip(), docs

# === Medical QA Logic ===
def medical_chat(user_message, history):
    # Add a temporary “loading” message
    history.append({"role": "assistant", "content": "⏳ Processing your query..."})
    yield history  # Show loading immediately
    
    # === Retrieve ICD-11 context ===
    context, docs = retrieve_icd_context(user_message, top_k=4)
    
    # === Guardrails ===
    if not context:
        final_answer = "I’m not sure, this information is not found in ICD-11."
    else:
        prompt = f"""
You are a strict ICD-11 medical assistant.
ONLY answer using the provided ICD-11 context.
If the answer is NOT in the context, reply exactly:
"I’m not sure, this information is not found in ICD-11."

ICD-11 Context:
{context}

Question: {user_message}

Answer:
"""
        raw_answer = llm.invoke(prompt)
        final_answer = raw_answer.strip()
    
    # === Sources ===
    sources = [doc.metadata.get("icd_code", "unknown") for doc in docs]
    if sources:
        final_answer += f"\n\n📚 *Based on ICD-11 entries:* {', '.join(set(sources))}"
    else:
        final_answer += "\n\n⚠️ *No exact ICD-11 match found.*"
    
    # === Remove loading + Add final ===
    history.pop()  # remove loading message
    history.append({"role": "user", "content": user_message})
    
    # === Stream response word by word ===
    partial = ""
    for word in final_answer.split():
        partial += word + " "
        # Temporarily show streaming text
        temp_hist = history + [{"role": "assistant", "content": partial.strip() + "▌"}]
        yield temp_hist
        time.sleep(0.02)
    
    # Finalize response
    history.append({"role": "assistant", "content": final_answer})
    yield history

# === Custom CSS for better UI ===
CUSTOM_CSS = """
.gradio-container {background: #0d1117 !important;}
.chatbot {height: 600px !important;}
.message.user {background: #1f6feb !important; color: white; border-radius: 12px; padding: 10px;}
.message.assistant {background: #161b22 !important; color: #c9d1d9; border-radius: 12px; padding: 10px;}
footer {display: none !important;}
"""

# === UI ===
with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 15px;">
        <h1 style="color:#58a6ff;"> Medical Assistant</h1>
        <p style="color:#8b949e;">AI-powered WHO ICD-11 Reference • Always cite official sources</p>
    </div>
    """)
    
    chatbot = gr.Chatbot(label="ICD-11 Agent", type="messages", height=500)
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask a medical question",
            label="Your Question",
            scale=8
        )
        send_btn = gr.Button(" Ask", scale=1)
        clear_btn = gr.Button(" Clear Chat", scale=1)
    
    # === Bind Logic ===
    send_btn.click(medical_chat, [user_input, chatbot], [chatbot])
    user_input.submit(medical_chat, [user_input, chatbot], [chatbot])
    clear_btn.click(lambda: [], None, chatbot)

demo.launch(server_name="localhost", server_port=7860)
