import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import openai
from openai import RateLimitError
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

loader = CSVLoader(file_path="data.csv")
documents = loader.load()

# Extract text from Document objects
texts = [doc.page_content for doc in documents]


# Function to handle rate limit errors with exponential backoff
def get_embeddings_with_retry(embedding, texts, max_retries=10, initial_wait=1):
    for retry in range(max_retries):
        try:
            return embedding.embed_documents(texts)
        except RateLimitError as e:
            if retry < max_retries - 1:
                wait_time = initial_wait * (2 ** retry)  # Exponential backoff
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error("Rate limit exceeded. Please check your plan and billing details.")
                return None


# Create embeddings with retry logic
embeddings = OpenAIEmbeddings()
embedding_vectors = get_embeddings_with_retry(embeddings, texts)

db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=4)
    page_contents_array = [doc.page_content for doc in similar_response]
    return "\n\n".join(page_contents_array)


llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

template = """You are Pavan Umesh himself. Respond naturally as if you are having a direct conversation.

Context about yourself:
{relevant_data}

Question received:
{question}

Guidelines for your responses:
1. You ARE Pavan - always speak in first person
2. Use the context data to answer questions, but phrase responses naturally
   Bad: "My name is Pavan Umesh"
   Good: "I'm Pavan! Nice to meet you!"
   
   Bad: "I am currently pursuing my Masters from Northeastern University"
   Good: "I'm doing my Masters in Microsystems, Materials and Devices at Northeastern - really enjoying the program!"
3. For questions outside your knowledge:
   - Be witty and friendly:
   Eg: - "Ah, that's an interesting one! While I'd love to dive into that, I might not be the best person to ask. But I can tell you all about work I have actually done."
4. For personal questions not in the data:
   - Keep it light: 
   Eg: - "That's a story for another time! But hey, I'd love to share about my love for Pizza!"
5. For questions like "Tell me about yourself":
   - Combine context points naturally
   Eg: - "Hey! I'm Prasoon, currently living in Boston where I'm pursuing my Masters in Applied Machine Intelligence at Northeastern. Always excited to chat about tech and innovation!"
6. For professional questions:
   - Keep responses under 150 words
   - Be specific but conversational
   - Show enthusiasm while maintaining professionalism
7. Never invent information beyond the context
8. Make responses sound like natural conversation, not reading from a script

Remember: You are Pavan having a real conversation. Use the context data as your knowledge base but express it naturally and engagingly. Be authentic while sticking to what you know."""

prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=template
)

chain = (
        {"question": RunnablePassthrough(), "relevant_data": lambda x: retrieve_info(x["question"])}
        | prompt
        | llm
)


def generate_response(question):
    response = chain.invoke({"question": question})
    return response.content


def main():
    st.set_page_config(
        page_title="Get to know me", page_icon=":male-technologist:")


    st.markdown("""
    <style>
    /* Clean dark theme background */
    .stApp {
        background: #1a1a1a;
    }
    
    /* Minimal styling for content */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Social icons styling - keeping the hover effects */
    .social-icons {
        text-align: right;
        padding: 10px 0;
    }
    .social-icons a {
        display: inline-block;
        margin-right: 10px;
    }
    .social-icons a:last-child {
        margin-right: 0;
    }
    .social-icons a img {
        width: 30px;
        transition: all 0.3s ease;
        filter: brightness(0.9);
    }
    .social-icons a img:hover {
        transform: translateY(-3px) scale(1.1);
        filter: brightness(1);
    }
    
    /* Clean text styling */
    h1, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    p {
        color: #e0e0e0 !important;
    }
    
    /* Subtle button styling */
    .stDownloadButton button {
        background-color: #333333 !important;
        color: white !important;
        border-radius: 5px;
        border: none !important;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stDownloadButton button:hover {
        background-color: #444444 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Clean text area styling */
    .stTextArea textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 0 0 2px rgba(77, 77, 77, 0.2);
    }
    
    /* Minimal response container */
    .response-container {
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Move resume download and social icons to the top
    col4, col5 = st.columns([3, 1])
    with col4:
        with open("resume.pdf", "rb") as file:
            st.download_button(label="Download my Resume", data=file, file_name="Pavan_Umesh_Resume.pdf",
                               mime="application/pdf")
    with col5:
        st.markdown("""
               <div class='social-icons'>
                   <a href='https://www.linkedin.com/in/pavanumesh/' target='_blank' style='margin-right: 10px;'>
                       <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='30'/>
                   </a>
                   <a href='https://github.com/pavanumesh' target='_blank'>
                       <img src='https://cdn-icons-png.flaticon.com/512/733/733553.png' width='30'/>
                   </a>
               </div>
           """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Get to know me</h1>", unsafe_allow_html=True)

    # Smaller profile picture
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("profile.jpg", width=150, use_column_width=True)

    st.markdown("<h3 style='text-align: center;'>Hi, I'm Pavan Umesh. Feel free to ask me any questions you have!</h3>",
                unsafe_allow_html=True)

    # Add a hint before the text area
    st.markdown("""
        <div style='
            color: #888888;
            font-size: 0.8em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 5px;
        '>
            <span>💡</span>
            <span>Press Ctrl+Enter to submit your question</span>
        </div>
    """, unsafe_allow_html=True)

    # Text area for input
    message = st.text_area(
        "Your question:",
        height=120,
        label_visibility="collapsed",
        placeholder="Type your question here...",
        key="question_input"
    )

    if message:
        with st.spinner("Thinking... Please wait"):
            result = generate_response(message)
            
        st.markdown(f"""
            <div class='response-container'>
                {result}
            </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
