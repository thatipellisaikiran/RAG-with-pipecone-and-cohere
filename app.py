import os
import time
import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere, CohereEmbeddings

# Set API keys (you can also use Streamlit's secrets management for security)
os.environ["COHERE_API_KEY"] = " YOUR API KEY "
os.environ['PINECONE_API_KEY'] = " YOUR API KEY "

# Streamlit App
st.title("PDF Querying App using LangChain and Cohere")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    try:
        # Read the PDF file as a byte stream
        pdf_data = uploaded_file.read()
        pdf_reader = fitz.open(stream=BytesIO(pdf_data), filetype="pdf")

        # Extract text from each page
        documents = []
        for page_num in range(pdf_reader.page_count):
            page = pdf_reader.load_page(page_num)
            text = page.get_text("text")
            documents.append(text)

        # Close the PDF reader
        pdf_reader.close()

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text("\n".join(documents))

        # Set up Pinecone
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        index_name = "docs-rag-chatbot"

        # Initialize Pinecone and create index if necessary
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=384,  # Adjust dimensionality if needed
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until the index is ready
            with st.spinner("Setting up Pinecone index..."):
                while not pc.describe_index(index_name).status["ready"]:
                    time.sleep(1)

        # Create embeddings using Cohere
        embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

        # Store documents in Pinecone
        vectorstore_from_docs = PineconeVectorStore.from_texts(
            docs,
            index_name=index_name,
            embedding=embeddings
        )

        # Initialize the Cohere LLM and RetrievalQA
        llm = ChatCohere()

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_from_docs.as_retriever()
        )

        # Input query from user
        query = st.text_input("Enter your query:")

        if query:
            # Retrieve and display the answer
            with st.spinner("Fetching answer..."):
                result = qa.run(query)
                st.write(f"**Answer**: {result}")

    except ValueError as e:
        st.error(f"Error: {e}")
        st.write("Please check your API keys and index setup.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.write("Please upload a PDF document to get started.")
