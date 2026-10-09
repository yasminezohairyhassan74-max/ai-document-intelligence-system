import html
import json
import logging
import os

import streamlit as st

import services
from llm import get_model
from rag import (DocumentError, add_document, format_reference, get_doc_info,
                 get_embedding, list_documents)
from settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR

logger = logging.getLogger("document_intelligence")
ALL_DOCS = "All documents"
RISK_DISCLAIMER = "AI analysis based on the document text. It is not legal or financial advice."

st.set_page_config(page_title="AI Document Intelligence", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.hero {padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #4f46e5, #7c3aed); color: white;}
.hero h1 {margin: 0; font-size: 1.8rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92;}
.ref-card {border-left: 4px solid #4f46e5; padding: .45rem .8rem; margin: .4rem 0;
           background: rgba(79, 70, 229, .08); border-radius: 6px; font-size: .9rem;}
.ref-card small {opacity: .85;}
</style>
<div class="hero">
  <h1>📄 AI Document Intelligence</h1>
  <p>Upload documents, ask questions with references, summarize, find risks, and compare.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the AI models... the first run takes a few minutes.")
def load_models():
    get_model()
    get_embedding()
    return True


load_models()


st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("processed", set())
st.session_state.setdefault("upload_errors", {})
st.session_state.setdefault("reports", {})
st.session_state.setdefault("compare_result", None)


def reset_uploader():
    # a new key gives an empty uploader
    st.session_state.uploader_key += 1
    st.session_state.processed = set()
    st.session_state.upload_errors = {}


def render_references(references):
    if not references:
        return
    with st.expander(f"📚 References ({len(references)})"):
        for ref in references:
            st.markdown(
                f'<div class="ref-card"><b>{html.escape(format_reference(ref))}</b><br>'
                f'<small>{html.escape(ref["snippet"])}</small></div>',
                unsafe_allow_html=True,
            )


def handle_uploads(uploaded_files):
    for uploaded in uploaded_files or []:
        key = (uploaded.name, uploaded.size)
        if key in st.session_state.processed:
            continue
        st.session_state.processed.add(key)

        safe_name = os.path.basename(uploaded.name)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        path = os.path.join(UPLOAD_DIR, safe_name)
        try:
            with open(path, "wb") as file:
                file.write(uploaded.getbuffer())
            with st.sidebar, st.spinner(f"Indexing {safe_name}..."):
                add_document(path)
            st.toast(f"✅ {safe_name} is ready", icon="📄")
        except DocumentError as error:
            st.session_state.upload_errors[safe_name] = str(error)
        except Exception:
            logger.exception("Unexpected error while indexing %s", safe_name)
            st.session_state.upload_errors[safe_name] = (
                f"Something went wrong while processing '{safe_name}'. Please try another file."
            )
        finally:
            if os.path.exists(path):
                os.remove(path)


def render_sidebar():
    st.sidebar.title("📁 Documents")
    st.sidebar.caption(f"{', '.join(ALLOWED_EXTENSIONS)} · up to {MAX_FILE_SIZE_MB} MB each")

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=[ext.strip(".") for ext in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    handle_uploads(uploaded_files)

    for message in st.session_state.upload_errors.values():
        st.sidebar.error(message)

    docs = list_documents()
    st.sidebar.divider()
    if not docs:
        st.sidebar.info("No documents yet. Upload a file to start.")
        return ALL_DOCS, docs

    st.sidebar.markdown(f"**{len(docs)} document(s) loaded**")
    scope = st.sidebar.selectbox("Active document", [ALL_DOCS] + docs)

    if scope != ALL_DOCS and st.sidebar.button("🗑️ Delete selected document", use_container_width=True):
        services.delete_document(scope)
        st.session_state.reports.pop(scope, None)
        st.session_state.compare_result = None
        reset_uploader()
        st.rerun()

    if st.sidebar.button("🧹 Clear session", use_container_width=True):
        services.clear_session()
        st.session_state.messages = []
        st.session_state.reports = {}
        st.session_state.compare_result = None
        reset_uploader()
        st.rerun()

    return scope, docs


def render_overview(scope, docs):
    names = docs if scope == ALL_DOCS else [scope]
    infos = [get_doc_info(name) for name in names]
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(names))
    col2.metric("Pages / sections", sum(i["pages"] for i in infos))
    col3.metric("Words", f"{sum(i['words'] for i in infos):,}")


def need_one_document(scope):
    if scope == ALL_DOCS:
        st.info("Select one document in the sidebar (Active document) to use this section.")
        return True
    return False


def tab_chat(scope, docs):
    if st.button("Clear chat"):
        services.reset_chat()
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("rewritten"):
                st.caption(f"Understood as: {message['rewritten']}")
            if message.get("not_found"):
                st.warning("This information was not found in the selected document(s).")
            render_references(message.get("references"))

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return
    if not docs:
        st.warning("Upload a document first.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    doc_name = None if scope == ALL_DOCS else scope
    with st.spinner("Searching the documents and thinking..."):
        try:
            result = services.chat(question, doc_name=doc_name)
            reply = {
                "role": "assistant",
                "content": result["answer"],
                "references": result["references"],
                "not_found": not result["found"],
                "rewritten": result["standalone_question"] if result["standalone_question"] != question else None,
            }
        except DocumentError as error:
            reply = {"role": "assistant", "content": str(error), "error": True}
        except Exception:
            logger.exception("Chat failed")
            reply = {"role": "assistant", "error": True,
                     "content": "Sorry, something went wrong while answering. Please try again."}
    st.session_state.messages.append(reply)
    st.rerun()


def tab_summary(scope):
    if need_one_document(scope):
        return
    if st.button("Generate summary", type="primary"):
        with st.spinner("Summarizing..."):
            try:
                services.summarize_document(scope)
            except Exception:
                logger.exception("Summary failed")
                st.error("Could not summarize this document. Please try again.")
                return
    if scope in services.summaries:
        st.subheader(scope)
        st.write(services.summaries[scope])


def tab_report(scope):
    if need_one_document(scope):
        return
    if st.button("Analyze document", type="primary"):
        with st.spinner("Analyzing... this can take a minute."):
            try:
                st.session_state.reports[scope] = services.generate_report(scope)
            except Exception:
                logger.exception("Report failed")
                st.error("Could not analyze this document. Please try again.")
                return

    report = st.session_state.reports.get(scope)
    if not report:
        st.info("Click “Analyze document” to get the summary, key points, and risks.")
        return
    if not report["parsed_ok"]:
        st.warning("The model's answer could not be structured, so only the summary is shown.")

    st.subheader("Summary")
    st.write(report["summary"])

    st.subheader("Key Points")
    for point in report["key_points"] or ["(none found)"]:
        st.markdown(f"- {point}")

    st.subheader("Potential Risks")
    st.warning(RISK_DISCLAIMER)
    for risk in report["risks"] or ["(none found)"]:
        st.markdown(f"- {risk}")

    render_references(report["references"])
    st.download_button("⬇️ Download report (JSON)", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name=f"{scope}_report.json", mime="application/json")


def tab_compare(docs):
    if len(docs) < 2:
        st.info("Upload at least two documents to compare them.")
        return
    col_a, col_b = st.columns(2)
    name_a = col_a.selectbox("Document A", docs, key="compare_a")
    name_b = col_b.selectbox("Document B", docs, index=1, key="compare_b")
    focus = st.text_input("Focus on a topic (optional)", placeholder="e.g. payment terms")

    if st.button("Compare", type="primary"):
        with st.spinner("Comparing..."):
            try:
                text = services.compare_documents(name_a, name_b, focus.strip() or None)
                st.session_state.compare_result = {"a": name_a, "b": name_b, "focus": focus.strip(), "text": text}
            except DocumentError as error:
                st.error(str(error))
            except Exception:
                logger.exception("Comparison failed")
                st.error("Could not compare these documents. Please try again.")

    result = st.session_state.compare_result
    if result:
        topic = f" · topic: {result['focus']}" if result["focus"] else ""
        st.subheader(f"{result['a']}  vs  {result['b']}{topic}")
        st.write(result["text"])
        st.caption("AI-generated comparison. Verify important details in the original documents.")


scope, docs = render_sidebar()

if docs:
    render_overview(scope, docs)

tabs = st.tabs(["💬 Chat", "📝 Summary", "🔑 Key Points & Risks", "⚖️ Compare"])
with tabs[0]:
    tab_chat(scope, docs)
with tabs[1]:
    tab_summary(scope)
with tabs[2]:
    tab_report(scope)
with tabs[3]:
    tab_compare(docs)
