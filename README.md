# 🧠 Project Wellness: AI-Powered Digital Mental Health Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Stateful-success)

## 📌 Overview
**Project Wellness** is an intelligent, scalable conversational agent designed for digital mental health support. 

Built as my university thesis, this project solves the critical limitations of standard LLM implementations (such as context amnesia, semantic drift, and hallucinations) by introducing a **Hybrid Stateful Architecture**. It bridges the gap between the generative capabilities of Large Language Models and the deterministic safety required in HealthTech applications.

---

##  Key Engineering Highlights 

Most chatbot wrappers rely on simple "Sliding Window" history or basic RAG, leading to high latency, token waste, and context conflicts. This project takes an engineering-first approach:

* **O(1) Token Optimization via Structured State Memory:** Instead of feeding the entire chat history back to the API, the system uses a proprietary `Memory Manager`. It extracts psychometric data (mood, sleep, topic) into an ephemeral JSON State Object. This keeps token usage flat (O(1)) and completely eliminates Context Conflict.
* **Dynamic Mode Blending (Prompt Injection):** The system does not use a static persona. It dynamically re-orchestrates the `System Prompt` in real-time based on the user's emotional state, seamlessly transitioning from *Expressive Empathy* to *Directive Cognitive Re-framing*.
* **Deterministic Safety Guardrails:** AI unpredictability is a liability in wellness apps. A robust, rule-based interception layer (using Regex pattern matching) scans user input *before* the API call. If high-risk language is detected, the LLM is short-circuited, and an immediate, hardcoded clinical fallback is triggered.
* **Explainable AI (XAI) Layer:** Features an integrated developer trace layer that exposes the backend decision-making process (mode weights, trigger paths, memory state) in real-time.
* **CBT-based Thought Reframe Studio:** A specialized, deterministic module guiding users through cognitive distortion labeling and psychoeducation, guaranteeing zero hallucinations.

---

##  Technology Stack
* **Backend Logic & NLP Processing:** Python 3
* **Frontend & Session Management:** Streamlit (Reactive UI, Ephemeral Session State)
* **Inference Engine:** OpenAI API (GPT-4o-mini)
* **Design Pattern:** Separation of Concerns (UI, State Manager, and LLM orchestration are strictly decoupled).

Install dependencies:
**Bash**
pip install -r requirements.txt

3. **Set your API Key:**
   Create a `.env` file or use Streamlit secrets to add your OpenAI API Key:
   ```env
   OPENAI_API_KEY="your-api-key-here"

4. **Run the application:**
   streamlit run app.py

---

##  Quick Start / Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/GEODPL/project-wellness.git](https://github.com/GEODPL/project-wellness.git)
   cd project-wellness

   ## 🎯 Let's Connect!
I am a passionate software engineer/researcher specializing in **Applied AI, Prompt Engineering, and Python Development**. I am actively looking for opportunities to bring my skills in building safe, scalable, and intelligent AI systems to a forward-thinking team.

📫 **Reach out to me:**
* **LinkedIn:** [www.linkedin.com/in/georgia-maria-diplou-247906336]
* **Email:** [diplougeorgiamaria@gmail.com]

> *Note: Project Wellness is a functional prototype developed for academic research and is not a substitute for professional medical advice.*
