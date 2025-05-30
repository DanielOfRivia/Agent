# Applicant Assistant Chatbot

This project is a **chatbot-based assistant** designed to help applicants to the **National University of Kyiv-Mohyla Academy (NaUKMA)** get accurate, helpful information about the admissions process. The assistant is powered by a **large language model** and enhanced with custom tools that retrieve answers from verified sources like the university's Telegram channel and official website.

## 🎯 Purpose

The chatbot was built to:
- Reduce the workload of admissions volunteers.
- Provide consistent and verified responses to frequently asked questions.
- Support prospective students with inclusive, easy-to-access information.

## 🧠 How It Works (Overview)

The chatbot uses:
- A **retrieval-augmented generation (RAG)** approach to pull real data from custom-built knowledge bases.
- Multiple tools including:
  - Telegram and website search
  - Discipline and tuition tools
  - Passing score retriever
- A user-friendly **web interface** built with Streamlit.
- **OpenAI's GPT-3.5-turbo** via API for natural conversation.

To understand the full architecture, components, and implementation details, **please refer to the thesis document** included in this repository:

📘 `Thesis-UKR.pdf` — Original thesis in Ukrainian

📘 `Thesis-US(translated)` — Version translated to English


## 📚 Contents

The thesis covers:
- Agent architecture and tool descriptions
- Prompt design and hallucination avoidance
- Experiments with embedding models and document splitting
- Quantitative and qualitative testing
- Future improvements (fine-tuning, RLHF, etc.)



For further details or technical questions, consult the thesis document or contact the author.
