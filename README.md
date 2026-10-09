 # 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Yasmine Zohairy Hassan Zohairy       |
| Project Name     | AI Document Intelligence System      |
| GitHub Username  | yasminezohairyhassan74-max           |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**I chose this project because I wanted to understand how RAG works in practice.**

This project is a system that helps you work with your documents using AI. You upload a file such as a contract, a report, or a lecture, and then you can ask questions about it. The answers come from the file itself, and each answer shows where the information was found (file name, page number, and chunk number).
The system can also summarize a document, list its key points and risks, and compare two documents.

### Why this project?
Reading long documents takes time, and it is easy to miss important details like amounts, deadlines, or penalties. A normal language model has never seen your files, so if you ask it about them it may make up an answer.

To solve this, the project uses **RAG (Retrieval-Augmented Generation)**. Instead of letting the model answer from its own memory, the system first searches the document for the parts related to the question, and then gives only those parts to the model. If the answer is not in the document, the system says that it could not find it.

### How it works
**1. When a file is uploaded**
```text
Upload file → Check the file → Read the text → Split it into chunks
→ Convert each chunk to an embedding → Store the embeddings in FAISS
```
**2. When a question is asked**
```text
Question → Search FAISS for the 3 closest chunks → Send the chunks and the question to the model
→ The model writes the answer → Show the answer with its references
```
**3. When a report is requested**
```text
Document summary + chunks about risks → Model → Output Parser
→ Summary, Key Points, Risks, References
```
### Where it can be used
* **Contracts:** find the duration, fees, penalties, and termination conditions
* **Reports:** find the main results
* **Study material:** ask about a lecture and see which page the answer is from
* **Two versions of a document:** compare them and see the differences
---
# ✨ Features

### 📂 Documents
* Upload one or more files (PDF, TXT, DOCX)
* The system checks every file and shows a clear message if it is empty, too large, damaged, unsupported, or a scanned document with no text
* Delete a single document or clear the whole session

### 💬 Question Answering
* Ask questions about one document or about all uploaded documents
* Every answer comes with references: file name, page number (for PDF), and chunk number
* If the answer is not in the document, the system says so instead of making one up
* Follow-up questions work: after "What is the total project fee?" you can ask "How is it paid?"

### 📝 Summary
* Generates a summary of a document
* Long documents are summarized section by section, then the summaries are combined

### 🔑 Key Points and Risks
* Produces a structured report: Summary, Key Points, Risks, and References
* The report can be downloaded as a JSON file
* If the model's answer cannot be parsed, the system shows the summary only instead of failing
* Risks are labeled as AI analysis, not legal or financial advice

### ⚖️ Document Comparison
* Compare two documents in general, or only on a topic you choose (for example "payment terms")
* The result is divided into similarities, differences, and a conclusion

### 🖥️ Interface
* Built with Streamlit: a sidebar for the files, and tabs for Chat, Summary, Key Points & Risks, and Compare
* Warnings and errors are shown in different colors from normal answers
---

# 🛠️ Technologies Used

* **Python**
* **Mistral-7B-Instruct-v0.2** (4-bit quantized, Hugging Face Transformers): the language model that writes the answers
* **LangChain**: prompts, chains, the Output Parser, document loaders, and the text splitter
* **Sentence Transformers** (`all-MiniLM-L6-v2`): turns text into embeddings
* **FAISS**: the vector database used to search the documents
* **Streamlit**: the user interface
* **Google Colab** (T4 GPU): where the project runs

### What came from the course
The language model, RAG (loading PDFs, splitting text, embeddings, and FAISS search), the chains, and the Output Parser.

### What I added
The Streamlit interface, support for several files (PDF, TXT, DOCX), references with page numbers, follow-up questions, document comparison, and a second try when the model's output cannot be parsed.

### A few choices
* I used a chunk size of 1000 and an overlap of 100, as in the course. I did not test other values.
* Each document has its own FAISS database, so answers from different files do not get mixed.
* References are taken from the document's metadata, so page numbers are never made up by the model.
* The prompt tells the model to answer only from the document, and to say so when it cannot find the answer.
---

# ⚙️ Installation

The language model needs a GPU, so the easiest way to run the project is **Google Colab** with the free T4 GPU.

### Run on Google Colab
1. Open the notebook in Colab: [Run_Streamlit_Colab.ipynb](https://colab.research.google.com/github/yasminezohairyhassan74-max/ai-document-intelligence-system/blob/main/Run_Streamlit_Colab.ipynb)
2. Go to **Runtime → Change runtime type** and choose **T4 GPU**.
3. Run the cells from top to bottom. The notebook installs the libraries and creates the project files.
4. The last cell prints a link that ends with `trycloudflare.com`. Open it to see the app.

The first run takes a few minutes because the model has to be downloaded. Keep the last cell running while you use the app.

### Run on your own computer
This needs an NVIDIA GPU.
```bash
git clone https://github.com/yasminezohairyhassan74-max/ai-document-intelligence-system.git
cd ai-document-intelligence-system
pip install -r requirements.txt
streamlit run app.py
```

### Project files
| File | What it does |
|---|---|
| `app.py` | The Streamlit interface |
| `services.py` | Question answering, chat, summary, comparison, and report |
| `chains.py` | The prompts, the chains, and the Output Parser |
| `rag.py` | Checking and loading files, splitting, embeddings, FAISS, and references |
| `llm.py` | Loads Mistral-7B and the `generate_text` function |
| `settings.py` | The settings (model name, chunk size, file limits) |
| `requirements.txt` | The Python libraries |
| `Run_Streamlit_Colab.ipynb` | Runs the app on Colab |
---

# 🚀 Usage

### 1. Upload documents
Upload one or more files (PDF, TXT, DOCX) from the sidebar. You can try the files in `sample_documents/`. Each file is checked, and a clear message is shown if it is empty, too large, damaged, unsupported, or a scanned document with no text. After indexing, the app shows the number of documents, pages, and words.

### 2. Choose the active document
Use **Active document** in the sidebar to work with one document, or keep **All documents** to search all of them. You can delete the selected document or clear the whole session from the sidebar.

### 3. 💬 Chat
Ask a question about your documents. The answer is written from the document text, and a **References** box shows the file name, page number (PDF only), chunk number, and a short snippet. Follow-up questions work: after "What is the total project fee?" you can ask "How is it paid?". If the answer is not in the document, the app says so and shows a warning instead of making one up.

### 4. 📝 Summary
Select **one** document and click **Generate summary**. Long documents are summarized section by section, then combined.

### 5. 🔑 Key Points & Risks
Select **one** document and click **Analyze document**. The app shows the summary, key points, and potential risks, with references. You can download the report as a JSON file. The risks are AI analysis, not legal or financial advice.

### 6. ⚖️ Compare
Upload at least two documents, choose Document A and Document B, and click **Compare**. Leave the topic empty for a general comparison, or type a topic such as "payment terms" to compare only that part. The result is divided into Similarities, Differences, and Conclusion.

### Example questions (sample contracts)
* What is the total project fee?
* What is the late payment fee?
* Does the agreement renew automatically?
* Who is the CEO? *(not in the file, so the app should say it could not find it)*

### Notebook version (without Streamlit)
[notebooks/Document_Intelligence_Notebook.ipynb](notebooks/Document_Intelligence_Notebook.ipynb)

---

# 📸 Demo

🎥 **Demo video:** [Watch the demo](https://drive.google.com/file/d/1nqWQ3RCAVJXc7sMJ2tnf-OEMv-S79CG9/view?usp=drive_link)

### 📂 Upload documents
Files are checked and indexed, then the app shows the number of documents, pages, and words.

![Upload documents](screenshots/load_files.png)

### 💬 Chat with references
The answer comes from the document, with the file name, page, and chunk. If the answer is not in the file, the app says so.

![Question 1](screenshots/ask_question.png)
![Question 2](screenshots/ask_question1.png)
![Question 3](screenshots/ask_question2.png)

### 📝 Summary
![Summary](screenshots/summary.png)

### 🔑 Key Points & Risks
![Analysis](screenshots/analysis.png)
![Analysis 2](screenshots/analysis1.png)

### ⚖️ Compare
![Compare](screenshots/compare.png)

More examples are in the [`examples/`](examples/) folder.

---

# 📈 Results

The system was tested with the two sample service agreements in `sample_documents/` (Delta Software Experts and Nile Software Solutions). Screenshots of each test are in the **Demo** section.

### What worked

| Feature | Result |
|---|---|
| **File upload** | Both PDF files were checked, read, and indexed. The app showed the number of documents, pages, and words. |
| **Question answering** | Questions about fees, late payment penalties, and renewal were answered from the document, with file name, page number, and chunk number. |
| **Not found in the document** | For a question that is not in the file, the app replied that it could not find the answer and showed a warning, with no references. |
| **Follow-up questions** | A short follow-up such as "How is it paid?" was understood from the previous question. |
| **Summary** | A summary was generated for a single document. |
| **Key Points & Risks** | The report showed the summary, key points, and risks, with references, and it was downloaded as a JSON file. |
| **Comparison** | The two contracts were compared, and the result showed the differences in fees, renewal, liability, and notice period. |

### Example (from the sample contracts)

| Question | Expected answer from the file |
|---|---|
| What is the total project fee? | Delta: 600,000 EGP in four installments. Nile: 450,000 EGP in three installments. |
| What is the late payment fee? | Delta: 3% per week after 10 days. Nile: 2% per week after 15 days. |
| Does the agreement renew automatically? | Delta: no. Nile: yes, for successive 12-month terms. |

### Limitations

* The model (Mistral-7B, 4-bit) runs on a free T4 GPU, so answers can take some time.
* The app keeps its state in memory, so it is meant for one user at a time.
* Very long documents are summarized from 6 sampled sections, not the whole file.
* Retrieval returns 3 chunks per document.
* Scanned PDFs (images only) are rejected because there is no OCR.
* Answers can change slightly between runs because the model uses sampling.
* The report is AI analysis, not legal or financial advice. Important details should be checked in the original document.

---

# 🔮 Future Improvements


* **OCR for scanned documents:** scanned PDFs (images only) are rejected now. Adding OCR would let the system read them.
* **Test chunk size and overlap:** I used 1000 and 100 as in the course, without testing other values. Trying different values and measuring the retrieval quality could improve the answers.
* **More stable answers:** lower the temperature (or turn off sampling) for question answering, so the same question gives the same answer.
* **Better Arabic support:** use a multilingual embedding model and test the system with Arabic documents.
* **Summarize the whole document:** long documents are now summarized from 6 sampled sections. Summarizing every section would cover the full file.
* **Separate session for each user:** the state is kept in memory and shared, so a database or per-user session would allow several users at the same time.
* **Faster responses:** try a lighter or faster model, or a hosted API.

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
