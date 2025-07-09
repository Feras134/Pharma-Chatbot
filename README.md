# Medication Info Chatbot

This is a Streamlit-based chatbot that provides clear, detailed, and patient-friendly summaries for medications. It fetches official drug label information from the **openFDA API** and uses **Google FLAN-T5** (via Hugging Face Transformers) to generate a human-readable explanation.

---

## Features

- **Search by Drug Name** (e.g., Advil, Ibuprofen)
- **Generates Explanations** of:
  - What the drug is used for
  - How to take it (dosage)
  - Possible side effects
- Uses **FLAN-T5** for natural language generation
- Based on **real FDA data**
- Web-based interface using **Streamlit**

---
