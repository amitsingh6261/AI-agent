import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Data Science Tutor",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# ==================================================
# 3. CHECK API KEY
# ==================================================

if not api_key:

    st.error(
        "❌ OPENROUTER_API_KEY not found."
    )

    st.info(
        "Please create a .env file and add your API key."
    )

    st.stop()


# ==================================================
# 4. CREATE OPENROUTER CLIENT
# ==================================================

client = OpenAI(

    base_url="https://openrouter.ai/api/v1",

    api_key=api_key
)


# ==================================================
# 5. SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# 6. SIDEBAR
# ==================================================

st.sidebar.title("🤖 AI Data Science Tutor")

st.sidebar.write(
    "Learn Data Science with AI"
)


mode = st.sidebar.selectbox(

    "🧠 Select Mode",

    [
        "AI Tutor",
        "Code Assistant",
        "CSV Data Analyzer"
    ]
)


# ==================================================
# 7. AI TUTOR MODE
# ==================================================

if mode == "AI Tutor":

    st.title("📚 AI Data Science Tutor")

    st.write(
        "Ask questions about Python, Pandas, "
        "Machine Learning, AI and Data Science."
    )


    # ----------------------------------------------
    # Topic
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Difficulty
    # ----------------------------------------------

    difficulty = st.sidebar.selectbox(

        "🎯 Difficulty",

        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )


    # ----------------------------------------------
    # Language
    # ----------------------------------------------

    language = st.sidebar.selectbox(

        "🌐 Language",

        [
            "Simple English",
            "Hinglish",
            "Hindi"
        ]
    )


    # ----------------------------------------------
    # Clear Chat
    # ----------------------------------------------

    if st.sidebar.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


    # ----------------------------------------------
    # Display Chat History
    # ----------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # ----------------------------------------------
    # User Question
    # ----------------------------------------------

    question = st.chat_input(
        "Ask your Data Science question..."
    )


    # ----------------------------------------------
    # AI Response
    # ----------------------------------------------

    if question:

        # Show user question

        with st.chat_message("user"):

            st.markdown(question)


        # Save question

        st.session_state.messages.append(

            {
                "role": "user",
                "content": question
            }

        )


        # ------------------------------------------
        # System Prompt
        # ------------------------------------------

        system_prompt = f"""

You are a friendly Data Science tutor.

Topic:
{topic}

Difficulty:
{difficulty}

Language:
{language}


Instructions:

1. Explain concepts in very simple language.

2. Give real-world examples.

3. Give Python code when useful.

4. Explain code step-by-step.

5. If the user is a beginner,
   avoid complicated terminology.

6. For Machine Learning,
   explain the complete flow:

   Data
   ↓
   Preprocessing
   ↓
   Training
   ↓
   Model
   ↓
   Prediction
   ↓
   Evaluation

7. Give practical examples.

8. If the user asks for interview questions,
   provide interview-style answers.

"""


        # ------------------------------------------
        # Prepare Messages
        # ------------------------------------------

        messages = [

            {
                "role": "system",
                "content": system_prompt
            }

        ]

        messages.extend(
            st.session_state.messages
        )


        # ------------------------------------------
        # Call AI
        # ------------------------------------------

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


                    # Save answer

                    st.session_state.messages.append(

                        {
                            "role": "assistant",
                            "content": answer
                        }

                    )


                except Exception as e:

                    st.error(
                        "❌ AI request failed."
                    )

                    st.write(e)


# ==================================================
# 8. CODE ASSISTANT
# ==================================================

elif mode == "Code Assistant":

    st.title("💻 AI Code Assistant")

    st.write(
        "Use AI to explain, debug and improve your Python code."
    )


    # ----------------------------------------------
    # Code Input
    # ----------------------------------------------

    code = st.text_area(

        "🐍 Enter Python Code",

        height=300,

        placeholder=
        "Example:\n\n"
        "import pandas as pd\n"
        "df = pd.read_csv('data.csv')\n"
        "print(df.head())"

    )


    # ----------------------------------------------
    # Action
    # ----------------------------------------------

    action = st.selectbox(

        "What should AI do?",

        [
            "Explain Code",
            "Find Error",
            "Improve Code",
            "Generate Example"
        ]

    )


    # ----------------------------------------------
    # Button
    # ----------------------------------------------

    if st.button(
        "🤖 Analyze Code"
    ):


        if code.strip() == "":

            st.warning(
                "⚠️ Please enter some Python code."
            )


        else:

            prompt = f"""

You are an expert Python and Data Science tutor.

User wants:
{action}


Python code:

{code}


Explain everything in simple language.

If there is an error:

1. Explain the error.
2. Explain why it happened.
3. Give the corrected code.
4. Explain the corrected code.

If improving code:

1. Show original problem.
2. Show improved code.
3. Explain improvements.

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
                        "❌ AI request failed."
                    )

                    st.write(e)


# ==================================================
# 9. CSV DATA ANALYZER
# ==================================================

elif mode == "CSV Data Analyzer":

    st.title("📊 AI CSV Data Analyzer")

    st.write(
        "Upload your CSV file and analyze your dataset."
    )


    # ----------------------------------------------
    # Upload File
    # ----------------------------------------------

    uploaded_file = st.file_uploader(

        "📂 Upload CSV File",

        type=["csv"]

    )


    if uploaded_file is None:

        st.info(
            "👆 Please upload a CSV file to start analysis."
        )


    else:

        try:

            # Read CSV

            df = pd.read_csv(
                uploaded_file
            )


            st.success(
                "✅ CSV uploaded successfully!"
            )


            # ======================================
            # DATASET OVERVIEW
            # ======================================

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


            # ======================================
            # DATA PREVIEW
            # ======================================

            st.subheader(
                "👀 Data Preview"
            )


            st.dataframe(

                df.head(10),

                use_container_width=True

            )


            # ======================================
            # DATA TYPES
            # ======================================

            st.subheader(
                "📋 Column Information"
            )


            info_df = pd.DataFrame(

                {

                    "Column":
                    df.columns,

                    "Data Type":
                    df.dtypes.astype(str).values,

                    "Missing Values":
                    df.isnull()
                    .sum()
                    .values,

                    "Unique Values":
                    df.nunique()
                    .values

                }

            )


            st.dataframe(

                info_df,

                use_container_width=True

            )


            # ======================================
            # MISSING VALUES
            # ======================================

            st.subheader(
                "❌ Missing Values"
            )


            missing = (
                df.isnull()
                .sum()
            )


            missing = missing[
                missing > 0
            ]


            if len(missing) == 0:

                st.success(
                    "🎉 No missing values found!"
                )

            else:

                missing_df = pd.DataFrame(

                    {
                        "Column":
                        missing.index,

                        "Missing Count":
                        missing.values

                    }

                )


                st.dataframe(
                    missing_df,
                    use_container_width=True
                )


            # ======================================
            # STATISTICS
            # ======================================

            st.subheader(
                "📈 Statistical Summary"
            )


            st.dataframe(

                df.describe(),

                use_container_width=True

            )


            # ======================================
            # NUMERICAL COLUMNS
            # ======================================

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
                    "📊 Data Visualization"
                )


                selected_column = st.selectbox(

                    "Select Column",

                    numerical_columns

                )


                chart_type = st.selectbox(

                    "Select Chart Type",

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

                        ax.set_ylabel(
                            selected_column
                        )


                    st.pyplot(fig)


            else:

                st.warning(
                    "No numerical columns found."
                )


            # ======================================
            # AI DATASET INSIGHTS
            # ======================================

            st.subheader(
                "🧠 AI Dataset Insights"
            )


            if st.button(
                "🤖 Analyze Dataset with AI"
            ):


                dataset_info = f"""

Columns:
{list(df.columns)}


Shape:
{df.shape}


Data Types:
{df.dtypes.to_string()}


Missing Values:
{df.isnull().sum().to_string()}


Statistics:
{df.describe().to_string()}


Sample Data:
{df.head(10).to_string()}

"""


                prompt = f"""

You are a professional Data Scientist.

Analyze the following dataset:

{dataset_info}


Explain in simple English:

1. What this dataset contains.

2. Important columns.

3. Missing values.

4. Interesting patterns.

5. Possible data quality problems.

6. Useful insights.

7. What analysis can be performed next.

8. Possible Machine Learning ideas.

Do not invent information.
Only use information supported by the dataset.

"""


                with st.spinner(
                    "🧠 AI is analyzing dataset..."
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


                        st.markdown(
                            answer
                        )


                    except Exception as e:

                        st.error(
                            "❌ AI request failed."
                        )

                        st.write(e)


            # ======================================
            # ASK YOUR DATA
            # ======================================

            st.subheader(
                "🔍 Ask Questions About Your CSV"
            )


            st.write(
                "Ask questions about your dataset in normal language."
            )


            user_question = st.text_input(

                "Example: What is the average salary?"

            )


            if st.button(
                "🔎 Ask About Data"
            ):


                if user_question.strip() == "":

                    st.warning(
                        "Please enter a question."
                    )


                else:

                    # Give AI complete data
                    # for smaller datasets

                    if len(df) <= 1000:

                        data_for_ai = (
                            df.to_string(
                                index=False
                            )
                        )

                    else:

                        data_for_ai = (
                            df.head(1000)
                            .to_string(
                                index=False
                            )
                        )


                    prompt = f"""

You are a Data Science assistant.

Here is the dataset:

{data_for_ai}


User Question:

{user_question}


Answer using the dataset.

Rules:

1. Use simple language.

2. Do not invent information.

3. If exact calculation is possible,
   provide the calculation.

4. Explain the answer.

5. If useful, provide Pandas code.

"""


                    with st.spinner(
                        "🔍 Searching your data..."
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
                                "📚 Answer"
                            )


                            st.markdown(
                                answer
                            )


                        except Exception as e:

                            st.error(
                                "❌ AI request failed."
                            )

                            st.write(e)


        except Exception as e:

            st.error(
                "❌ Could not read CSV file."
            )

            st.write(e)