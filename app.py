import streamlit as st
import os
import time
import whisper
import warnings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress annoying whisper warnings
warnings.filterwarnings("ignore")

# 1. Setup
load_dotenv()
client = genai.Client()

# 2. Schemas
class KeyTerm(BaseModel):
    term: str
    definition: str

class ConceptDetail(BaseModel):
    concept_name: str
    detailed_explanation: str 
    examples: list[str]       

class StudyNotes(BaseModel):
    comprehensive_summary: str
    key_terms: list[KeyTerm]
    core_concepts: list[ConceptDetail]

# --- PHASE 9: CACHING & RESILIENCE ---
@st.cache_data(show_spinner=False)
def process_transcript(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    chunks = splitter.split_text(raw_text)
    
    all_chunk_notes = []
    
    # Define our backup plans (Models 3.5 and above!)
    fallback_models = ['gemini-3.5-flash', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
    max_retries = 3
    
    for i, chunk in enumerate(chunks):
        prompt = f"""
        You are an expert university professor creating an exhaustive study guide.
        Read the following transcript chunk and extract the information in depth.
        RULES: Do NOT invent information. Only use the facts provided.
        --- TRANSCRIPT CHUNK ---
        {chunk}
        """
        
        chunk_success = False
        
        # 1. THE RETRY LOOP (Try up to 3 times)
        for attempt in range(max_retries):
            if chunk_success:
                break # If we succeeded, break out of the retry loop!
                
            # 2. THE MODEL FALLBACK LOOP
            for model_name in fallback_models:
                try:
                    response = client.models.generate_content(
                        model=model_name, 
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=StudyNotes,
                        ),
                    )
                    all_chunk_notes.append(response.parsed)
                    chunk_success = True
                    time.sleep(4) # Standard rate limit pause
                    break # Success! Break out of the model loop
                    
                except Exception as e:
                    # Model failed. Print to terminal but don't crash the web app!
                    print(f"Warning: {model_name} failed on chunk {i+1}. Trying next model...")
                    time.sleep(1) 
            
            # If we looped through all models and STILL failed, wait 5 seconds before attempting again
            if not chunk_success:
                print(f"All models failed on attempt {attempt+1}. Waiting 5 seconds before retrying...")
                time.sleep(5)
                
        # If we tried 3 times across all models and it still failed, error out on the UI.
        if not chunk_success:
            st.error(f"Critical Error: Chunk {i+1} failed completely after all retries. The API might be down.")
            
    # Reduce Phase 
    master_summary = ""
    unique_terms_dict = {}
    unique_concepts_dict = {}

    for note in all_chunk_notes:
        master_summary += note.comprehensive_summary + "\n\n"
        for kt in note.key_terms:
            unique_terms_dict[kt.term] = kt.definition
        for concept in note.core_concepts:
            unique_concepts_dict[concept.concept_name] = concept
            
    return master_summary, unique_terms_dict, unique_concepts_dict

# --- CACHING THE AUDIO TRANSCRIPTION ---
@st.cache_data(show_spinner=False)
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# 3. UI Setup
st.title("🎓 Smart Notes Generator")
st.write("Upload a lecture audio file (.mp3, .wav) OR a text transcript (.txt)")

lecture_title = st.text_input("Enter Lecture Title:", "Computer Networks")
# Now accepting all three file types!
uploaded_file = st.file_uploader("Upload File", type=["txt", "mp3", "wav"])

if uploaded_file is not None:
    if st.button("Generate Study Guide"):
        
        # Grab the file extension (txt, mp3, or wav)
        file_extension = uploaded_file.name.split(".")[-1].lower()
        raw_text = ""
        
        # --- THE WORKFLOW ROUTING ---
        if file_extension in ["mp3", "wav"]:
            with st.spinner("🎧 Audio detected! Transcribing with Whisper (This takes a moment)..."):
                
                # Save the uploaded RAM file to the hard drive so Whisper can read it
                os.makedirs("data/raw_audio", exist_ok=True)
                temp_audio_path = f"data/raw_audio/{uploaded_file.name}"
                
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Transcribe!
                raw_text = transcribe_audio(temp_audio_path)
                st.success("Audio transcribed successfully!")
                
        elif file_extension == "txt":
            with st.spinner("📄 Text detected! Reading file..."):
                raw_text = uploaded_file.read().decode("utf-8")
                
        # --- THE AI PROCESSING ---
        with st.spinner("🧠 AI is analyzing the lecture..."):
            master_summary, unique_terms_dict, unique_concepts_dict = process_transcript(raw_text)
            
        # --- DISPLAY RESULTS ---
        st.subheader("📝 Comprehensive Summary")
        st.write(master_summary)

        st.subheader("🔑 Key Terms")
        for term, definition in unique_terms_dict.items():
            st.markdown(f"**{term}**: {definition}")

        st.subheader("📌 Core Concepts")
        for name, concept in unique_concepts_dict.items():
            st.markdown(f"### {name}")
            st.write(concept.detailed_explanation)
            if concept.examples:
                st.markdown("**Examples:**")
                for ex in concept.examples:
                    st.markdown(f"- {ex}")
                    
        st.success("✨ Notes Generated Successfully!")