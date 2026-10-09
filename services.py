from langchain_core.prompts import PromptTemplate

import rag
from chains import (as_list, clean_answer, compare_chain, extract_json_block, format_instructions,
                    map_chain, output_parser, qa_chain, reduce_chain, report_template, rewrite_chain)
from llm import generate_text
from rag import DocumentError, doc_label, make_references, retrieve
from settings import LLM_MAX_LENGTH

summaries = {}
chat_history = []


def delete_document(name):
    rag.remove_document(name)
    summaries.pop(name, None)


def clear_session():
    rag.clear_documents()
    summaries.clear()
    chat_history.clear()


def summarize_document(name, section_chars=3500, max_sections=6):
    if name in summaries:
        return summaries[name]

    text = rag.get_full_text(name)
    sections = [text[i:i + section_chars] for i in range(0, len(text), section_chars)]
    if len(sections) > max_sections:
        step = len(sections) / max_sections
        sections = [sections[int(i * step)] for i in range(max_sections)]

    if len(sections) == 1:
        summary = clean_answer(map_chain.run(sections[0]))
    else:
        partial = [clean_answer(map_chain.run(s)) for s in sections]
        joined = "\n\n".join(f"Part {i+1}: {s}" for i, s in enumerate(partial))
        summary = clean_answer(reduce_chain.run(joined))

    summaries[name] = summary
    return summary


def ask_question(query, doc_name=None, k=3):
    docs = retrieve(query, doc_name, k)
    context = "\n\n".join(f"[{doc_label(d)}]\n{d.page_content}" for d in docs)
    answer = clean_answer(qa_chain.run({"context": context, "question": query}))
    not_found_phrases = ("could not find", "does not mention", "not mentioned", "no information")
    found = not any(phrase in answer.lower() for phrase in not_found_phrases)
    return {
        "question": query,
        "answer": answer,
        "found": found,
        "references": make_references(docs) if found else [],
    }


def chat(query, doc_name=None):
    standalone = query
    if chat_history:
        history = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:])
        rewritten = clean_answer(rewrite_chain.run({"history": history, "question": query}))
        standalone = rewritten.split("\n")[0].strip()
        if standalone.count('"') >= 2:
            standalone = standalone.split('"')[1].strip()
        standalone = standalone or query
    result = ask_question(standalone, doc_name)
    result["question"] = query
    result["standalone_question"] = standalone
    chat_history.append((query, result["answer"]))
    return result


def reset_chat():
    chat_history.clear()


def compare_documents(name_a, name_b, focus=None):
    if name_a == name_b:
        raise DocumentError("Please choose two different documents.")
    if focus:
        text_a = "\n".join(d.page_content for d in retrieve(focus, name_a, k=3))
        text_b = "\n".join(d.page_content for d in retrieve(focus, name_b, k=3))
    else:
        text_a, text_b = summarize_document(name_a), summarize_document(name_b)
    result = compare_chain.run({"name_a": name_a, "text_a": text_a,
                                "name_b": name_b, "text_b": text_b})
    return clean_answer(result)


def generate_report(name):
    summary = summarize_document(name)
    risk_docs = retrieve("risks penalties liability termination obligations deadlines fines", name, k=4)
    risk_context = "\n\n".join(f"({doc_label(d)}) {d.page_content}" for d in risk_docs)

    prompt = PromptTemplate(
        template=report_template,
        input_variables=["summary", "risk_context", "format_instructions"]
    ).format(summary=summary, risk_context=risk_context, format_instructions=format_instructions)

    data, parsed_ok = None, False
    # try twice, the model sometimes returns a broken JSON
    for _ in range(2):
        response = generate_text(prompt, max_length=LLM_MAX_LENGTH)
        try:
            data = output_parser.parse(extract_json_block(response))
            parsed_ok = True
            break
        except Exception:
            data = None

    if data is None:
        data = {"summary": summary, "key_points": [], "risks": []}

    return {
        "document": name,
        "summary": str(data.get("summary", summary)).strip(),
        "key_points": as_list(data.get("key_points")),
        "risks": as_list(data.get("risks")),
        "references": make_references(risk_docs),
        "parsed_ok": parsed_ok,
    }
