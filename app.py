import streamlit as st
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# 3. UI Setup
st.title("🎓 Smart Notes Generator")
st.write("Upload a lecture transcript (.txt) to generate a comprehensive study guide.")

lecture_title = st.text_input("Enter Lecture Title:", "Computer Networks")
uploaded_file = st.file_uploader("Upload Transcript", type="txt")

# 4. The Action Button
if uploaded_file is not None:
    if st.button("Generate Study Guide"):
        
        # st.spinner shows a loading wheel while our slow code runs!
        with st.spinner("Reading and Chunking text..."):
            raw_text = uploaded_file.read().decode("utf-8")
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            chunks = splitter.split_text(raw_text)
            
            st.success(f"Created {len(chunks)} chunks!")

        with st.spinner("AI is analyzing the lecture (This takes a moment)..."):
            all_chunk_notes = []
            
            # MAP PHASE
            for chunk in chunks:
                prompt = f"""
                You are an expert university professor creating an exhaustive study guide.
                Read the following transcript chunk and extract the information in depth.
                RULES: Do NOT invent information. Only use the facts provided.
                
                --- TRANSCRIPT CHUNK ---
                {chunk}
                """
                try:
                    response = client.models.generate_content(
                        model='gemini-3.5-flash', 
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=StudyNotes,
                        ),
                    )
                    all_chunk_notes.append(response.parsed)
                    time.sleep(4) # Rate limit protection
                except Exception as e:
                    st.error(f"Error on chunk: {e}")
            
            # REDUCE PHASE
            master_summary = ""
            unique_terms_dict = {}
            unique_concepts_dict = {}

            for note in all_chunk_notes:
                master_summary += note.comprehensive_summary + "\n\n"
                for kt in note.key_terms:
                    unique_terms_dict[kt.term] = kt.definition
                for concept in note.core_concepts:
                    unique_concepts_dict[concept.concept_name] = concept

        # 5. Display the Results dynamically on the Webpage!
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