import os
import sys
from unittest.mock import MagicMock

# --- 1. Mock pkg_resources in memory to satisfy CrewAI Telemetry ---
try:
    import pkg_resources
except ImportError:
    mock_pkg = MagicMock()
    mock_pkg.get_distribution.return_value.version = "0.80.0"
    mock_pkg.DistributionNotFound = Exception
    sys.modules["pkg_resources"] = mock_pkg

# Disable CrewAI telemetry to avoid unnecessary background requests
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# --- 2. Core Imports ---
import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, LLM

# --- 3. Page Configuration ---
st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 4. Helper Functions ---
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text from an uploaded PDF file in-memory using pypdf."""
    try:
        reader = PdfReader(uploaded_file)
        extracted_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

        cleaned_text = extracted_text.strip()
        if not cleaned_text:
            raise ValueError(
                "No readable text found in this PDF. It appears to be an image-only scan "
                "or an empty file. Please use a text-based PDF or paste your resume text manually."
            )
        return cleaned_text
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")


def initialize_crewai_agent(api_key: str, model_name: str) -> Crew:
    """Initializes the single-agent, single-task CrewAI pipeline."""
    if "groq" in model_name.lower():
        os.environ["GROQ_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = api_key

    llm = LLM(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
    )

    resume_auditor = Agent(
        role="Objective Technical Resume Auditor",
        goal="Accurately and critically evaluate candidate resumes against target job descriptions without making assumptions.",
        backstory=(
            "You are a strict, objective corporate talent evaluator. You review resumes "
            "with zero bias and zero extrapolation. "
            "CRITICAL ACCURACY RULE: You MUST NEVER invent, assume, or infer experiences, "
            "skills, certifications, metrics, or education that are not explicitly documented "
            "in the candidate's resume text. If a requirement is not clearly stated, you must "
            "categorize it as 'Missing' or 'Unclear / Not Demonstrated'. You provide candid, "
            "constructive guidance to help candidates present their genuine achievements truthfully."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    review_task = Task(
        description=(
            "Carefully review the provided Candidate Resume against the Target Job Description.\n\n"
            "=== CANDIDATE RESUME ===\n"
            "{resume_text}\n\n"
            "=== TARGET JOB DESCRIPTION ===\n"
            "{job_description}\n\n"
            "EVALUATION RULES:\n"
            "1. Ground every statement solely in the provided text. Never assume unlisted credentials.\n"
            "2. Identify clear matches, explicit omissions, and unproven claims.\n"
            "3. Format your assessment into the exact 9 sections outlined below."
        ),
        expected_output=(
            "Generate a professional, structured Markdown report with these exact section headings:\n\n"
            "## 1. Match Summary\n"
            "Concise, objective summary of the candidate's alignment with the role based only on stated facts.\n\n"
            "## 2. Skills Found\n"
            "Bullet list of required technical and domain skills explicitly verified in the resume.\n\n"
            "## 3. Missing Requirements\n"
            "Mandatory job requirements, technologies, or capabilities completely absent from the resume.\n\n"
            "## 4. Unclear / Not Demonstrated\n"
            "Items mentioned vaguely or lacking measurable evidence, context, or project application.\n\n"
            "## 5. Experience Gaps\n"
            "Differences between stated experience levels (years, scope, seniority) and the job demands.\n\n"
            "## 6. Education & Qualification Gaps\n"
            "Discrepancies in degrees, certifications, or required credentials.\n\n"
            "## 7. Resume Improvements\n"
            "Direct, honest recommendations to better showcase existing, verifiable qualifications.\n\n"
            "## 8. Keywords to Consider\n"
            "Relevant job description terms the candidate can legitimately include if they possess the skill.\n\n"
            "## 9. Priority Action Plan\n"
            "Numbered list of 3-5 immediate steps the candidate should take to improve their application."
        ),
        agent=resume_auditor,
    )

    return Crew(
        agents=[resume_auditor],
        tasks=[review_task],
        verbose=False,
    )


# --- 5. UI Header & Privacy Notice ---
st.title("📄 Single-Agent Resume Reviewer")
st.markdown(
    "Evaluate candidate resumes against real job requirements with zero hallucinations. "
    "Every gap and match is determined strictly from the supplied text."
)

st.info(
    "🔒 **Privacy Notice:** Resumes and job descriptions are processed in-memory and are never "
    "stored or logged in any database. The text is submitted to your configured LLM API provider "
    "solely to generate this analysis."
)

# --- 6. Configuration Check ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")
api_key = groq_api_key or openai_api_key

configured_model = st.secrets.get(
    "MODEL",
    "openai/gpt-oss-120b" if groq_api_key else "gpt-4o-mini",
)

if not api_key:
    st.error(
        "⚠️ **API Key Missing!** Please configure `GROQ_API_KEY` (or `OPENAI_API_KEY`) "
        "and `MODEL` in your Streamlit Cloud Settings > Secrets."
    )

# --- 7. Application Inputs ---
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("1. Candidate Resume")
    resume_input_method = st.radio(
        "Choose input format:",
        ["Upload PDF", "Paste Plain Text"],
        horizontal=True,
    )

    resume_text = ""
    if resume_input_method == "Upload PDF":
        uploaded_pdf = st.file_uploader(
            "Upload Resume (PDF format)",
            type=["pdf"],
            help="Upload a digital PDF. Scanned images requiring OCR are not supported.",
        )
        if uploaded_pdf:
            try:
                resume_text = extract_text_from_pdf(uploaded_pdf)
                st.success(f"Extracted {len(resume_text.split())} words from PDF.")
            except ValueError as ve:
                st.error(f"⚠️ {str(ve)}")
    else:
        resume_text = st.text_area(
            "Paste Resume Text",
            height=320,
            placeholder="Paste raw resume text here (Work history, Education, Technical skills)...",
        )

with col2:
    st.subheader("2. Target Job Description")
    job_description = st.text_area(
        "Paste Job Description",
        height=370,
        placeholder="Paste complete job post, including responsibilities, prerequisites, and nice-to-haves...",
    )

# --- 8. Review Action & Processing ---
st.divider()

if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
    if not api_key:
        st.error("Cannot proceed: Missing API key. Check your Streamlit secrets.")
    elif not resume_text.strip():
        st.warning("Please provide a resume by uploading a valid PDF or pasting text.")
    elif not job_description.strip():
        st.warning("Please paste the target job description to run the comparison.")
    else:
        with st.spinner("Analyzing resume against job requirements..."):
            try:
                crew = initialize_crewai_agent(api_key, configured_model)
                result = crew.kickoff(
                    inputs={
                        "resume_text": resume_text.strip(),
                        "job_description": job_description.strip(),
                    }
                )

                st.success("Analysis Complete!")
                st.markdown(str(result))

                st.download_button(
                    label="📥 Download Review Report (.md)",
                    data=str(result),
                    file_name="resume_match_review.md",
                    mime="text/markdown",
                )

            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    st.error(
                        "⚠️ **Rate Limit Exceeded:** The AI provider is temporarily busy or rate-limited. "
                        "Please wait a few seconds and try again."
                    )
                elif "authentication" in error_msg or "401" in error_msg or "invalid api key" in error_msg:
                    st.error(
                        "⚠️ **Authentication Failed:** Invalid API Key. Please verify your credentials "
                        "in your Streamlit App Secrets."
                    )
                elif "timeout" in error_msg:
                    st.error(
                        "⚠️ **Request Timed Out:** The inference server took too long to respond. "
                        "Try again with a more concise job description."
                    )
                else:
                    st.error(
                        f"⚠️ **Execution Error:** Unable to complete review. Details: {str(e)}"
                    )
