import streamlit as st
import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration


@st.cache_resource
def load_model():
    model_id = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    return tokenizer, model


def get_medication_info(med_name):
    url = "https://api.fda.gov/drug/label.json"
    params = {"search": f"openfda.brand_name:{med_name}", "limit": 1}

    try:
        res = requests.get(url, params=params)
        data = res.json()
        if "results" not in data:
            return None

        result = data["results"][0]
        return {
            "uses": result.get("indications_and_usage", [""])[0],
            "dosage": result.get("dosage_and_administration", [""])[0],
            "side_effects": result.get("adverse_reactions", [""])[0]
        }
    except:
        return None


def generate_summary(tokenizer, model, med_name, info):
    prompt = f"""
You are a medical assistant. Provide a detailed, patient-friendly summary of the following drug.

Drug Name: {med_name}

Uses: {info['uses']}
Dosage: {info['dosage']}
Side Effects: {info['side_effects']}

Explain this in clear, non-technical English.
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    st.set_page_config(page_title=" Medication Chatbot", page_icon="")
    st.title(" Medication Info Chatbot")
    st.markdown("Enter the name of a medication and get a detailed, patient-friendly explanation.")

    tokenizer, model = load_model()
    med_name = st.text_input("Enter a medication name:")

    if st.button("Get Information") and med_name:
        with st.spinner(" Fetching from openFDA..."):
            info = get_medication_info(med_name)

        if info:
            with st.spinner(" Generating summary..."):
                summary = generate_summary(tokenizer, model, med_name, info)
            st.subheader(" Medication Summary")
            st.write(summary)
        else:
            st.error(" Medication not found or no data available.")


if __name__ == "__main__":
    main()
