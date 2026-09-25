import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# 1. PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Data Science Tutor",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# 2. LOAD API KEY
# ==================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:

    st.error(
        "❌ OPENROUTER_API_KEY not found in .env file."
    )

    st.stop()


# ==================================================
# 3. OPENROUTER CLIENT
# ==================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ==================================================
# 4. SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# 5. SIDEBAR
# ==================================================

st.sidebar.title("🤖 AI Data Science Tutor")

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "AI Tutor",
        "Code Assistant",
        "CSV Data Analyzer"
    ]
)


# ==================================================
# 6. AI TUTOR
# ==================================================

if mode == "AI Tutor":

    st.title("📚 AI Data Science Tutor")

    st.write(
        "Learn Python, Pandas, Machine Learning, "
        "AI and Data Science."
    )

    topic = st.sidebar.selectbox(
        "📚 Topic",
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

    if st.sidebar.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask your question..."
    )

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

Explain everything in simple language.

Give practical examples.

For coding questions provide Python code.

Explain important code step-by-step.

Do not use unnecessarily complicated words.

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

            with st.spinner(
                "🤖 AI is thinking..."
            ):

                try:

                    response = client.chat.completions.create(

                        model="openrouter/free",

                        messages=messages

                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        "❌ AI Error"
                    )

                    st.write(e)


# ==================================================
# 7. CODE ASSISTANT
# ==================================================

elif mode == "Code Assistant":

    st.title("💻 AI Code Assistant")

    code = st.text_area(
        "🐍 Enter Python Code",
        height=300
    )

    action = st.selectbox(
        "Select Action",
        [
            "Explain Code",
            "Find Error",
            "Improve Code",
            "Generate Example"
        ]
    )

    if st.button("🤖 Analyze Code"):

        if code.strip() == "":

            st.warning(
                "Please enter Python code."
            )

        else:

            prompt = f"""

You are an expert Python tutor.

Action:
{action}

Code:

{code}

Explain in simple language.

If there is an error:

1. Explain the error.
2. Explain why it happened.
3. Give corrected code.
4. Explain the correction.

"""

            with st.spinner(
                "🐍 Analyzing code..."
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

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    st.subheader(
                        "🤖 AI Result"
                    )

                    st.markdown(answer)

                except Exception as e:

                    st.error(
                        "❌ AI Error"
                    )

                    st.write(e)


# ==================================================
# 8. CSV ANALYZER
# ==================================================

elif mode == "CSV Data Analyzer":

    st.title("📊 AI CSV Data Analyzer")

    st.write(
        "Upload your CSV and perform real data analysis."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload CSV",
        type=["csv"]
    )


    if uploaded_file is None:

        st.info(
            "👆 Upload a CSV file to start."
        )


    else:

        try:

            df = pd.read_csv(
                uploaded_file
            )


            # ==========================================
            # OVERVIEW
            # ==========================================

            st.success(
                "✅ Dataset uploaded!"
            )

            st.subheader(
                "📌 Dataset Overview"
            )

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
                    int(
                        df.isnull()
                        .sum()
                        .sum()
                    )
                )

            with col4:

                st.metric(
                    "Duplicates",
                    int(
                        df.duplicated()
                        .sum()
                    )
                )


            # ==========================================
            # PREVIEW
            # ==========================================

            st.subheader(
                "👀 Data Preview"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )


            # ==========================================
            # COLUMN INFORMATION
            # ==========================================

            st.subheader(
                "📋 Column Information"
            )

            info = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type":
                        df.dtypes.astype(str).values,
                    "Missing":
                        df.isnull().sum().values,
                    "Unique":
                        df.nunique().values
                }
            )

            st.dataframe(
                info,
                use_container_width=True
            )


            # ==========================================
            # STATISTICS
            # ==========================================

            st.subheader(
                "📈 Statistical Summary"
            )

            st.dataframe(
                df.describe(),
                use_container_width=True
            )


            # ==========================================
            # VISUALIZATION
            # ==========================================

            numerical_columns = (
                df
                .select_dtypes(
                    include="number"
                )
                .columns
                .tolist()
            )


            if numerical_columns:

                st.subheader(
                    "📊 Visualization"
                )

                selected_column = st.selectbox(
                    "Select Numerical Column",
                    numerical_columns
                )

                chart_type = st.selectbox(
                    "Chart Type",
                    [
                        "Histogram",
                        "Box Plot"
                    ]
                )

                if st.button(
                    "📈 Generate Chart"
                ):

                    fig, ax = plt.subplots()

                    if chart_type == "Histogram":

                        ax.hist(
                            df[
                                selected_column
                            ].dropna(),
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

                    else:

                        ax.boxplot(
                            df[
                                selected_column
                            ].dropna()
                        )

                        ax.set_title(
                            f"{selected_column} Box Plot"
                        )

                    st.pyplot(fig)


            # ==========================================
            # REAL PANDAS ANALYSIS
            # ==========================================

            st.subheader(
                "🧠 Ask Your Data"
            )

            st.write(
                "Ask a question and Pandas will "
                "calculate the result."
            )


            question = st.text_input(
                "Example: Which city has the highest average salary?"
            )


            if st.button(
                "🔍 Analyze Question"
            ):

                if question.strip() == "":

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    # ----------------------------------
                    # Dataset information for AI
                    # ----------------------------------

                    columns = list(
                        df.columns
                    )

                    sample = df.head(
                        10
                    ).to_string(
                        index=False
                    )


                    prompt = f"""

You are a Python Pandas expert.

Dataset columns:

{columns}


Sample data:

{sample}


User question:

{question}


Your task:

Create ONE Pandas expression that can calculate
the answer from the dataframe named df.

Rules:

1. Use only Pandas.
2. Do not use external libraries.
3. Do not explain anything.
4. Return ONLY Python code.
5. The code must store the final answer
   in a variable named result.

Examples:

Question:
What is the average salary?

Answer:
result = df["Salary"].mean()


Question:
Which city has the highest average salary?

Answer:
result = df.groupby("City")["Salary"].mean().idxmax()

"""

                    with st.spinner(
                        "🧠 AI is creating Pandas calculation..."
                    ):

                        try:

                            response = client.chat.completions.create(

                                model="openrouter/free",

                                messages=[
                                    {
                                        "role":
                                            "system",
                                        "content":
                                            prompt
                                    }
                                ]

                            )

                            code_answer = (
                                response
                                .choices[0]
                                .message
                                .content
                            )


                            # ==================================
                            # CLEAN AI CODE
                            # ==================================

                            code_answer = (
                                code_answer
                                .replace(
                                    "```python",
                                    ""
                                )
                                .replace(
                                    "```",
                                    ""
                                )
                                .strip()
                            )


                            st.subheader(
                                "🐼 Pandas Calculation"
                            )

                            st.code(
                                code_answer,
                                language="python"
                            )


                            # ==================================
                            # EXECUTE PANDAS CODE
                            # ==================================

                            safe_globals = {
                                "pd": pd
                            }

                            safe_locals = {
                                "df": df
                            }


                            exec(
                                code_answer,
                                safe_globals,
                                safe_locals
                            )


                            result = safe_locals.get(
                                "result"
                            )


                            # ==================================
                            # SHOW RESULT
                            # ==================================

                            st.subheader(
                                "📊 Actual Result"
                            )


                            if isinstance(
                                result,
                                pd.Series
                            ):

                                st.dataframe(
                                    result,
                                    use_container_width=True
                                )

                            elif isinstance(
                                result,
                                pd.DataFrame
                            ):

                                st.dataframe(
                                    result,
                                    use_container_width=True
                                )

                            else:

                                st.success(
                                    str(result)
                                )


                            # ==================================
                            # AI EXPLANATION
                            # ==================================

                            explanation_prompt = f"""

You are a Data Science tutor.

User question:
{question}

Pandas calculation:
{code_answer}

Actual result:
{result}

Explain the result in very simple language.

Do not change the result.

"""

                            explanation_response = (
                                client
                                .chat.completions.create(

                                    model="openrouter/free",

                                    messages=[
                                        {
                                            "role":
                                                "system",
                                            "content":
                                                explanation_prompt
                                        }
                                    ]

                                )
                            )


                            explanation = (
                                explanation_response
                                .choices[0]
                                .message
                                .content
                            )


                            st.subheader(
                                "🤖 AI Explanation"
                            )

                            st.markdown(
                                explanation
                            )


                        except Exception as e:

                            st.error(
                                "❌ Could not calculate the answer."
                            )

                            st.write(
                                e
                            )


        except Exception as e:

            st.error(
                "❌ Could not read CSV."
            )

            st.write(
                e
            )