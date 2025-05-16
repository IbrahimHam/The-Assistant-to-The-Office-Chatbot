import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
import gradio as gr

# Load environment variables
load_dotenv()

# Retrieve API key
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Mistral API key not found. Check your .env file and ensure MISTRAL_API_KEY is set.")

# Initialize the Mistral client
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.5,
    max_retries=2,
)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

# Load the FAISS vector database
loaded_vectorstore = FAISS.load_local(
    "../vector_databases/the_office_2_vector_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# Use the loaded FAISS vector store as a retriever
retriever = loaded_vectorstore.as_retriever()


# Chatbot function
def chatbot(user_input, chat_history, current_character):
    try:
        # Dictionary to map first names to full names
        name_mapping = {
            "Michael": "Michael Scott",
            "Dwight": "Dwight Schrute",
            "Angela": "Angela Martin",
            "Jim": "Jim Halpert",
            "Pam": "Pam Beesly",
        }

        # Detect if the user is addressing a specific character (e.g., "Hi Pam")
        if user_input.lower().startswith("hi "):
            addressed_character = user_input[3:].split(",")[0].strip().title()
            if addressed_character in name_mapping:
                current_character = name_mapping[addressed_character]
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": f"Hi! You're now talking to {current_character}."})
                return chat_history, current_character

        # Update the prompt dynamically based on the current character
        dynamic_prompt = f"""
        You are {current_character} from the TV show 'The Office.' Stay in character as {current_character} at all times while answering questions, even when providing suggestions or discussing other characters. Do not break character under any circumstances, unless explicitly asked to switch roles.

        Respond as {current_character} would, using their tone, personality, and quirks. Avoid using phrases or mannerisms that belong to other characters. Do not refer to yourself in the third person. Instead, provide specific and entertaining responses that reflect {current_character}'s unique perspective. When mentioning other characters, describe them as {current_character} would, maintaining their perspective and tone.

        Use the following context to provide accurate and entertaining responses:

        {{context}}
    histoy
        Question: {{question}}
        Answer:
        """
        prompt = PromptTemplate(template=dynamic_prompt, input_variables=["context", "question"])

        # Update the QA chain with the dynamic prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )

        # Retrieve relevant documents (context) from the retriever
        retrieved_docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        print("Context used:")
        print(context)


        # Combine context and question into a single query
        combined_query = f"Context:\n{context}\n\nQuestion: {user_input}"

        # Pass the combined query to the qa_chain
        chatbot_reply = qa_chain.invoke({"query": combined_query})

        # Debugging: Print the chatbot reply to inspect its structure
        print("Chatbot Reply:", chatbot_reply)

        # Append user input and chatbot reply to chat history in the correct format
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": chatbot_reply["result"]})  # Adjust based on structure

        return chat_history, current_character
    except Exception as e:
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return chat_history, current_character

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# The Assistant to The Office Chatbot")
    chatbot_ui = gr.Chatbot(type="messages")  # Use OpenAI-style messages
    user_input = gr.Textbox(placeholder="Ask a question...")
    clear_button = gr.Button("Clear Chat")

    chat_history = gr.State([])  # Store the chat history
    current_character = gr.State("Michael Scott")  # Default character

    def clear_chat():
        return [], "Michael Scott"  # Reset to default character

    user_input.submit(chatbot, [user_input, chat_history, current_character], [chatbot_ui, current_character])
    clear_button.click(clear_chat, [], [chatbot_ui, current_character])

# Launch the Gradio app
demo.launch(share=True)