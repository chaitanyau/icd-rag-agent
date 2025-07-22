import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Load Vector DB 
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectordb = Chroma(
    persist_directory="embeddings",
    embedding_function=embedding
)

# Local LLM 
llm = OllamaLLM(model="phi3:medium")

#  Synonym map 
SYNONYM_MAP = {
    "heart attack": "myocardial infarction",
    "stroke": "cerebrovascular accident",
    "flu": "influenza",
    "chickenpox": "varicella",
    "whooping cough": "pertussis",
    "lockjaw": "tetanus",
    "german measles": "rubella",
}

# Retrieval
def retrieve_icd_context(query, top_k=4):
    """Retrieve top_k ICD-11 context chunks with semantic + fuzzy expansion."""

    normalized_query = query.lower().strip()

    # Expand synonyms if found
    for layman, medical in SYNONYM_MAP.items():
        if layman in normalized_query:
            normalized_query += " " + medical

    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(normalized_query)

    # If nothing retrieved → last attempt with only synonyms
    if not docs:
        for layman, medical in SYNONYM_MAP.items():
            if layman in query.lower():
                docs = retriever.invoke(medical)
                break

    if not docs:
        return "", []

    combined_context = "\n\n".join([doc.page_content for doc in docs])
    return combined_context.strip(), docs


def medical_chat(user_message, history):
    history = history or []

 
    context, docs = retrieve_icd_context(user_message, top_k=4)

    if not context or len(context) < 50:
        answer = " I couldn’t find an exact ICD-11 match for your query. Try rephrasing or be more specific."
    else:
        
        prompt = f"""
You are a WHO ICD-11 medical assistant.
- Use ONLY the provided ICD-11 context for accuracy.
- If the user’s query is slightly different but semantically related, explain the closest ICD-11 match.
- If there is *no relevant* ICD-11 entry, say:
"I’m not sure, this information is not found in ICD-11."

ICD-11 Context:
{context}

User Question: {user_message}

Answer in a clear, medical style:
"""
        raw_answer = llm.invoke(prompt)
        answer = raw_answer.strip()

    # Collect ICD source codes + clickable WHO links
    if docs:
        sources = [
            f"[{doc.metadata.get('icd_code', 'unknown')}]({doc.metadata.get('browser_url', '#')})"
            for doc in docs
        ]
        answer += "\n\n📚 **ICD-11 References:** " + ", ".join(sources)
    else:
        answer += "\n\n⚠️ *No exact ICD-11 match found.*"

    # Append to chat history (Gradio v4 → role-based format)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})

    return history, history

# Modern UI with Loader 
with gr.Blocks(theme=gr.themes.Soft(), css="""
#chatbox {height: 500px !important;}
#loading_circle {display:none; text-align:center;}
""") as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center;"> ICD-11 Medical Assistant</h1>
        <p style="text-align:center;">
        <b>Answers WHO ICD-11 definitions, symptoms, and conditions.</b>  
        <i>Now with semantic search (handles layman terms like "heart attack")</i>
        </p>
        """
    )

    chatbot = gr.Chatbot(label="ICD-11 Agent", elem_id="chatbox", type="messages")

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask a medical question (e.g. 'What is heart attack?')",
            label="Your Query"
        )
        submit_btn = gr.Button(" Ask")
        clear_btn = gr.Button(" Clear")

    # Circular loader while processing
    loading_circle = gr.Markdown(" *Processing your query...*", elem_id="loading_circle")

    # Submit event → show loader, run chat, then hide loader
    def show_loader():
        return gr.update(visible=True)

    def hide_loader():
        return gr.update(visible=False)

    submit_btn.click(show_loader, outputs=loading_circle).then(
        medical_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, chatbot]
    ).then(hide_loader, outputs=loading_circle)

    clear_btn.click(lambda: [], None, chatbot)

demo.launch(server_name="localhost", server_port=7860)
