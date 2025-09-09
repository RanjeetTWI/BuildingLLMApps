import streamlit as st
import requests

st.title("AI Story Generator 📖✨")

genre = st.selectbox("Choose a Genre", ["Fantasy", "Sci-Fi", "Mystery", "Adventure"])
character_input = st.text_input("Characters (comma separated or leave blank for auto)")
paragraphs = st.slider("Number of Paragraphs", 2, 10, 3)
audience = st.text_input("Audience (like Young or age group)")
tone = st.text_input("Audience (like dark and suspenseful")
include_images = st.selectbox("Want to Include Image?", ["No", "Yes"])
include_preface = st.selectbox("Want to Include Preface?", ["No", "Yes"])

if st.button("Generate Story"):
    payload = {
        "genre": genre,
        "characters": [c.strip() for c in character_input.split(",")] if character_input else None,
        "paragraphs": paragraphs,
        "audience": audience,
        "tone": tone,
        "include_images": include_images,
        "include_preface": include_preface,
    }
    response = requests.post("http://127.0.0.1:8000/generate_story", json=payload)
    data = response.json()

    st.subheader("Preface")
    st.write(data["preface"])

    for i, para in enumerate(data["story"]):
        st.markdown(f"### Section {i + 1}")
        st.write(para)
        # If images are generated, show them
        if "images" in data:
            st.image(data["images"][i])
