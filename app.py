
import streamlit as st
from pytube import YouTube
import whisper
import os
from utils import download_youtube_video, transcribe_video_to_text
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from IPython.display import Markdown

from dotenv import load_dotenv

load_dotenv()  # Load the environment variables from .env
api_key = os.getenv("MY_API_KEY")

# Set a default value for the YouTube URL in session state
st.title("YouTube Video Transcription and Q&A with LangChain Agent")

# Persistent YouTube URL and question query in session state
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ""

if 'query' not in st.session_state:
    st.session_state.query = ""

# YouTube video URL input
st.session_state.youtube_url = st.text_input("YouTube Video URL", st.session_state.youtube_url)

# Placeholders for documents and FAISS database
documents = []
faiss_db = None

# Transcription logic with Transcribe button
if st.session_state.youtube_url and st.button("Transcribe"):
    # Path to download the video
    output_dir = 'downloads'  # Ensure this directory exists
    video_path = os.path.join(output_dir, "video.mp4")

    # Download the YouTube video
    with st.spinner("Downloading video..."):
        video_path = download_youtube_video(st.session_state.youtube_url, output_dir)

    # Transcribe the video
    with st.spinner("Transcribing video..."):
        transcription = transcribe_video_to_text(video_path)

        # Save the transcription to a text file
        transcript_file_path = os.path.join(output_dir, "transcripts.txt")
        with open(transcript_file_path, 'w') as transcript_file:
            transcript_file.write(transcription)

        # Create Document and FAISS if not already created
        documents.append(Document(page_content=transcription, metadata={"source": "YouTube Transcription"}))

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        faiss_db = FAISS.from_documents(documents, embeddings)

        # Clean up the downloaded video
        if os.path.exists(video_path):
            os.remove(video_path)


loader = DirectoryLoader("./downloads/")
documents = loader.load_and_split()

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever,
    "search_exercise_docs",
    "Searches and return excerpts from notes, youtube transcripts and web articles about neural networks as well as my roadmap to learn."
)

tools = [tool]

prompt = hub.pull("hwchase17/openai-tools-agent")
llm_chat = ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0, openai_api_key=api_key)

agent = create_openai_tools_agent(llm_chat, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)
st.write("Ask questions related to the transcribed video content or other topics.")

# Query input and Submit button
st.session_state.query = st.text_input("Enter your question here:", st.session_state.query)
submit = st.button("Submit")  # Submit button for asking questions

# Use a spinner while processing the question
if agent_executor and st.session_state.query and submit:
    with st.spinner("Processing your question..."):
        result = agent_executor.invoke({"input": st.session_state.query})

        # Display the result
        if "output" in result:
            st.markdown(result["output"])
        else:
            st.warning("No results found. Try rewording your question or checking the data source.")


