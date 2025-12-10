import os
import re  # for cleaning abbreviation output

# Make Streamlit behave nicely on Windows
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup
from google import genai  # Google Gen AI SDK for Gemini API

# ------------- Streamlit page config ------------- #

st.set_page_config(
    page_title="LLM Document App (Q4 - Gemini)",
    page_icon="🤖",
    layout="wide",
)

# ------------- LLM loading (Google Gemini via API) ------------- #

@st.cache_resource
def load_llm():
    """
    Load a closed-source LLM (Google Gemini) via the Gemini API.
    Requires GEMINI_API_KEY in environment or Streamlit secrets.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create an API key in Google AI Studio "
            "and set it as an environment variable or in st.secrets."
        )

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.0-flash"

    def call_llm(prompt: str) -> str:
        """Send a text prompt to Gemini and return the response text."""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return getattr(response, "text", str(response))

    return call_llm

# ------------- File reading helpers ------------- #

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text

def read_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt(uploaded_file):
    data = uploaded_file.read()
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")

def read_html(uploaded_file):
    data = uploaded_file.read()
    soup = BeautifulSoup(data.decode("utf-8", errors="ignore"), "html.parser")
    return soup.get_text(separator="\n")

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if name.endswith(".docx"):
        return read_docx(uploaded_file)
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    if name.endswith(".html") or name.endswith(".htm"):
        return read_html(uploaded_file)

    return ""

# ------------- Helper: clean abbreviation output ------------- #

def clean_abbrev_answer(raw_answer: str) -> list[str]:
    """
    Keep only lines of the form:
    ABBR: full term
    """
    clean_lines = []
    for line in raw_answer.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-Z][A-Z0-9]{1,10})\s*:\s*(.+)$', line)
        if m:
            abbr = m.group(1).strip()
            full = m.group(2).strip()
            clean_lines.append(f"{abbr}: {full}")
    return clean_lines

# ------------- Streamlit UI ------------- #

def main():
    st.title("Input to AI (Q4 - Gemini)")

    st.write(
        "Ask a question and optionally upload documents. "
        "This version uses Google Gemini (closed-source LLM via API)."
    )

    question = st.text_area("Your question:", height=80)

    uploaded_files = st.file_uploader(
        "Upload files (optional):",
        type=["pdf", "docx", "txt", "html", "htm"],
        accept_multiple_files=True,
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
            return

        lower_q = question.lower()
        is_abbrev_task = (
            "abbreviation" in lower_q
            or "abbreviations" in lower_q
            or "abbrev" in lower_q
        )

        with st.spinner("Thinking..."):
            llm = load_llm()

            # ------ SPECIAL ABBREVIATION MODE ------ #
            if uploaded_files and is_abbrev_task:
                st.subheader("AI Response (Abbreviation Index):")

                for uploaded in uploaded_files:
                    doc_text = extract_text_from_file(uploaded)
                    if not doc_text:
                        continue

                    max_chars = 6000
                    if len(doc_text) > max_chars:
                        doc_text = doc_text[:max_chars]

                    prompt = (
                        "You are an assistant that EXTRACTS ABBREVIATIONS "
                        "from scientific documents.\n"
                        "Your task is to build an abbreviation index from the DOCUMENT below.\n"
                        "An abbreviation index is a list like:\n"
                        "WDC: weighted degree centrality\n"
                        "SH: structural holes\n"
                        "ERGM: exponential random graph model\n"
                        "\n"
                        "Rules:\n"
                        "- Only include abbreviations that actually appear in the document.\n"
                        "- Most abbreviations appear in the form 'full term (ABBR)'. Use that pattern.\n"
                        "- The left side must be just the abbreviation token (e.g., 'WDC', 'ERGM', 'CAS').\n"
                        "- The right side must be the full term from the document.\n"
                        "- Do NOT invent or guess abbreviations not supported by the text.\n"
                        "- Do NOT add explanations or extra sentences.\n"
                        "- Output ONLY the index, one abbreviation per line, in the format 'ABBR: full term'.\n"
                        "\n"
                        "[DOCUMENT]\n"
                        f"{doc_text}\n\n"
                        "[ANSWER]\n"
                    )

                    result = llm(prompt)

                    if "[ANSWER]" in result:
                        answer = result.split("[ANSWER]", 1)[-1].strip()
                    elif "Answer:" in result:
                        answer = result.split("Answer:", 1)[-1].strip()
                    else:
                        answer = result.strip()

                    abbrev_lines = clean_abbrev_answer(answer)

                    st.markdown(f"### File: `{uploaded.name}`")
                    if not abbrev_lines:
                        st.markdown("_No abbreviations found._")
                    else:
                        bullets = []
                        for line in abbrev_lines:
                            abbr, full = line.split(":", 1)
                            bullets.append(f"- **{abbr.strip()}**: {full.strip()}")
                        st.markdown("\n".join(bullets))

                return  # done with abbreviation mode

            # ------ Normal QA mode ------ #
            document_text = ""
            if uploaded_files:
                texts = []
                for uploaded in uploaded_files:
                    txt = extract_text_from_file(uploaded)
                    if txt:
                        texts.append(f"--- FILE: {uploaded.name} ---\n{txt}\n")
                document_text = "\n\n".join(texts)

            max_chars = 6000
            if len(document_text) > max_chars:
                document_text = document_text[:max_chars]

            if document_text:
                prompt = (
                    "You are a helpful assistant.\n"
                    "Use ONLY the document text below to answer the question.\n"
                    "If the answer is not in the document, say exactly:\n"
                    "'I am not sure based on the document.'\n"
                    "Use clear, short sentences.\n\n"
                    "[DOCUMENT]\n"
                    f"{document_text}\n\n"
                    "[QUESTION]\n"
                    f"{question}\n\n"
                    "[ANSWER]"
                )
            else:
                prompt = (
                    "You are a helpful assistant.\n"
                    "Answer the question clearly and stay on topic.\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )

            try:
                result = llm(prompt)

                if "[ANSWER]" in result:
                    answer = result.split("[ANSWER]", 1)[-1].strip()
                elif "Answer:" in result:
                    answer = result.split("Answer:", 1)[-1].strip()
                else:
                    answer = result.strip()

                st.subheader("AI Response:")
                st.code(answer, language="text")

                if document_text:
                    with st.expander("Show document text the model saw"):
                        st.text(document_text)

            except Exception as e:
                st.error(f"Error running the model: {e}")


if __name__ == "__main__":
    main()
