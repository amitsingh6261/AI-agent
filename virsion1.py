import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI


# ==============================
# 1. PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Data Science Tutor",
    page_icon="🤖",
    layout="wide"
)


# ==============================
# 2. LOAD API KEY
# ==============================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found in .env file.")
    st.stop()


# ==============================
# 3. OPENROUTER CLIENT
# ==============================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ==============================
# 4. SESSION STATE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==============================
# 5. SIDEBAR
# ==============================

st.sidebar.title("⚙️ Tutor Settings")

topic = st.sidebar.selectbox(
    "📚 Select Topic",
    [
        "Python",
        "Pandas",
        "NumPy",
        "Machine Learning",
        "Artificial Intelligence",
        "Statistics",
        "Data Science",
        "SQL",
        "DBMS"
    ]
)

difficulty = st.sidebar.selectbox(
    "🎯 Difficulty",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

language = st.sidebar.selectbox(
    "🌐 Language",
    [
        "Simple English",
        "Hinglish",
        "Hindi"
    ]
)

mode = st.sidebar.selectbox(
    "🧠 Learning Mode",
    [
        "Tutor",
        "Code Generator",
        "Code Debugger",
        "Quiz Generator",
        "Interview Mode",
        "Notes Generator"
    ]
)


# ==============================
# 6. CLEAR CHAT BUTTON
# ==============================

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ==============================
# 7. MAIN TITLE
# ==============================

st.title("🤖 AI Data Science Tutor")

st.write(
    "Learn Python, Pandas, Machine Learning, AI, "
    "Statistics and Data Science with AI."
)


# ==============================
# 8. SHOW CHAT HISTORY
# ==============================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==============================
# 9. USER QUESTION
# ==============================

question = st.chat_input(
    "Ask something about Data Science..."
)


# ==============================
# 10. AI RESPONSE
# ==============================

if question:

    # Show user message
    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ==============================
    # CREATE SYSTEM PROMPT
    # ==============================

    system_prompt = f"""
You are an expert and friendly Data Science tutor.

Selected Topic:
{topic}

Difficulty Level:
{difficulty}

Language:
{language}

Learning Mode:
{mode}

Rules:

1. Explain everything in very simple language.
2. Give practical examples.
3. If code is required, provide clean Python code.
4. Explain important code line by line.
5. Avoid unnecessary complicated words.
6. If the user is a beginner, assume they know basic Python only.
7. For Machine Learning questions, explain:
   Data → Model → Training → Prediction → Evaluation
8. For coding questions, show the expected output when useful.
9. For interview mode, behave like an interviewer.
10. For quiz mode, create useful MCQs.
11. For notes mode, provide proper structured notes.
12. For Hinglish, use Hindi + English naturally.
"""


    # ==============================
    # PREPARE MESSAGES
    # ==============================

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(
        st.session_state.messages
    )


    # ==============================
    # CALL AI
    # ==============================

    with st.chat_message("assistant"):

        with st.spinner("🤖 AI is thinking..."):

            try:

                response = client.chat.completions.create(

                    model="openrouter/free",

                    messages=messages

                )

                answer = response.choices[0].message.content

                st.markdown(answer)


                # Save AI response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                st.error(
                    "❌ Something went wrong while connecting to AI."
                )

                st.code(str(e))


# ==============================
# 11. FOOTER
# ==============================

st.sidebar.markdown("---")

st.sidebar.info(
    "🤖 AI Data Science Tutor\n\n"
    "Learn • Practice • Code • Analyze"
)