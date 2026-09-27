import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Data Science Tutor",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. LOAD API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not found in .env file.")
    st.stop()


# ==========================================
# 3. OPENROUTER CLIENT
# ==========================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ==========================================
# 4. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 5. SIDEBAR
# ==========================================

st.sidebar.title("🤖 AI Data Science Tutor")

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "AI Tutor",
        "Code Assistant",
        "CSV Data Analyzer"
    ]
)


# ==========================================
# 6. AI TUTOR MODE
# ==========================================

if mode == "AI Tutor":

    st.title("📚 AI Data Science Tutor")

    topic = st.sidebar.selectbox(
        "Select Topic",
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
        "Difficulty Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    language = st.sidebar.selectbox(
        "Language",
        [
            "Simple English",
            "Hinglish",
            "Hindi"
        ]
    )

    question = st.chat_input(
        "Ask something..."
    )

    # Show old messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    if question:

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        system_prompt = f"""
You are a friendly Data Science tutor.

Topic: {topic}

Difficulty: {difficulty}

Language: {language}

Rules:
- Explain in simple words.
- Give examples.
- Give Python code when useful.
- Explain code clearly.
- If possible give real-world examples.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(
            st.session_state.messages
        )

        with st.chat_message("assistant"):

            with st.spinner("🤖 AI is thinking..."):

                try:

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=messages
                    )

                    answer = response.choices[0].message.content

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error("AI Error")
                    st.write(e)


# ==========================================
# 7. CODE ASSISTANT
# ==========================================

elif mode == "Code Assistant":

    st.title("💻 AI Code Assistant")

    code = st.text_area(
        "Paste your Python code here:",
        height=250
    )

    action = st.selectbox(
        "What do you want AI to do?",
        [
            "Explain Code",
            "Find Errors",
            "Improve Code",
            "Generate Example"
        ]
    )

    if st.button("🤖 Analyze Code"):

        if code.strip() == "":

            st.warning("Please enter some code.")

        else:

            prompt = f"""
You are a Python and Data Science coding tutor.

User wants:
{action}

Here is the code:

{code}

Explain everything in simple language.
If there is an error, explain:
1. What is the error?
2. Why did it happen?
3. How to fix it?
4. Correct code.
"""

            with st.spinner("Checking code..."):

                try:

                    response = client.chat.completions.create(
                        model="openrouter/free",
                        messages=[
                            {
                                "role": "system",
                                "content": prompt
                            }
                        ]
                    )

                    answer = response.choices[0].message.content

                    st.subheader("🤖 AI Result")

                    st.markdown(answer)

                except Exception as e:

                    st.error("AI Error")
                    st.write(e)


# ==========================================
# 8. CSV DATA ANALYZER
# ==========================================

elif mode == "CSV Data Analyzer":

    st.title("📊 AI CSV Data Analyzer")

    st.write(
        "Upload a CSV file and analyze your dataset automatically."
    )


    # ------------------------------------------
    # Upload CSV
    # ------------------------------------------

    uploaded_file = st.file_uploader(
        "📂 Upload CSV File",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)


            # ------------------------------------------
            # Dataset Information
            # ------------------------------------------

            st.success("✅ Dataset uploaded successfully!")


            st.subheader("📌 Dataset Overview")


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Rows",
                    df.shape[0]
                )


            with col2:

                st.metric(
                    "Columns",
                    df.shape[1]
                )


            with col3:

                st.metric(
                    "Missing Values",
                    df.isnull().sum().sum()
                )


            with col4:

                st.metric(
                    "Duplicates",
                    df.duplicated().sum()
                )


            # ------------------------------------------
            # Data Preview
            # ------------------------------------------

            st.subheader("👀 Data Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )


            # ------------------------------------------
            # Column Information
            # ------------------------------------------

            st.subheader("📋 Column Information")

            info_df = pd.DataFrame({

                "Column": df.columns,

                "Data Type": df.dtypes.astype(str).values,

                "Missing Values": df.isnull().sum().values,

                "Unique Values": df.nunique().values

            })

            st.dataframe(
                info_df,
                use_container_width=True
            )


            # ------------------------------------------
            # Missing Values
            # ------------------------------------------

            st.subheader("❌ Missing Values")

            missing = df.isnull().sum()

            missing = missing[missing > 0]


            if len(missing) == 0:

                st.success(
                    "🎉 No missing values found!"
                )

            else:

                st.dataframe(
                    missing.rename("Missing Count")
                )


            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            st.subheader("📈 Statistical Summary")

            st.dataframe(
                df.describe(),
                use_container_width=True
            )


            # ------------------------------------------
            # Numerical Columns
            # ------------------------------------------

            numerical_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()


            if numerical_columns:

                st.subheader("📊 Create Chart")

                selected_column = st.selectbox(
                    "Select Numerical Column",
                    numerical_columns
                )


                chart_type = st.selectbox(
                    "Select Chart",
                    [
                        "Histogram",
                        "Box Plot"
                    ]
                )


                if st.button("📈 Generate Chart"):

                    fig, ax = plt.subplots()


                    if chart_type == "Histogram":

                        ax.hist(
                            df[selected_column].dropna(),
                            bins=20
                        )

                        ax.set_title(
                            f"{selected_column} Distribution"
                        )

                        ax.set_xlabel(
                            selected_column
                        )

                        ax.set_ylabel(
                            "Frequency"
                        )


                    elif chart_type == "Box Plot":

                        ax.boxplot(
                            df[selected_column].dropna()
                        )

                        ax.set_title(
                            f"{selected_column} Box Plot"
                        )

                        ax.set_ylabel(
                            selected_column
                        )


                    st.pyplot(fig)


            # ------------------------------------------
            # AI Data Insights
            # ------------------------------------------

            st.subheader("🧠 AI Data Insights")


            if st.button("🤖 Analyze Dataset with AI"):

                sample_data = df.head(10).to_string()

                summary = df.describe().to_string()

                prompt = f"""
You are a Data Science expert.

Analyze this dataset.

Dataset columns:
{list(df.columns)}

First 10 rows:
{sample_data}

Statistical summary:
{summary}

Give the answer in simple language.

Explain:

1. What this dataset represents.
2. Important columns.
3. Interesting patterns.
4. Possible problems in the data.
5. Important insights.
6. What analysis could be done next.
7. Possible Machine Learning ideas.

Do not invent information that cannot be supported by the dataset.
"""

                with st.spinner(
                    "🧠 AI is analyzing your dataset..."
                ):

                    try:

                        response = client.chat.completions.create(
                            model="openrouter/free",
                            messages=[
                                {
                                    "role": "system",
                                    "content": prompt
                                }
                            ]
                        )

                        answer = response.choices[0].message.content

                        st.markdown(answer)

                    except Exception as e:

                        st.error("AI Error")
                        st.write(e)


        except Exception as e:

            st.error(
                "❌ Could not read the CSV file."
            )

            st.write(e)