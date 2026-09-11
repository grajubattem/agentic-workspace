RAG Quality Gate
================

Project Overview
----------------
This project is a local Retrieval-Augmented Generation (RAG) application. A user
uploads PDF or Excel documents, asks a question, and receives an answer generated
from the uploaded content. The application also evaluates answer quality using
fast local checks and optional RAGAS LLM-as-judge metrics.

The application is designed to reduce unsupported answers by retrieving relevant
document content before asking the language model to answer.


Architecture
------------

	User uploads PDF/XLSX
			|
			v
	Extract text and tables
			|
			v
	Split documents into chunks
			|
			v
	Generate embeddings with Ollama
			|
			v
	Store vectors in persistent ChromaDB
			|
			v
	User asks a question
			|
			v
	Retrieve the most relevant chunks
			|
			v
	Generate an answer with the Ollama LLM
			|
			+-----------------------+
			|                       |
			v                       v
	Local quality checks       RAGAS evaluation
	relevance                  faithfulness
	groundedness               answer relevancy
							   context precision
							   context recall


End-to-End Workflow
-------------------

1. The Streamlit application accepts PDF, XLSX, and XLS uploads.

2. Uploaded files are saved in a temporary directory. A SHA-256 digest of the
   uploaded contents is used to create a unique Chroma collection name. This
   prevents different document sets from sharing an index.

3. PDF files are read page by page with pypdf. Excel workbooks are read sheet by
   sheet with pandas. Source names, page numbers, and sheet names are included in
   the extracted content and metadata.

4. Extracted documents are split with RecursiveCharacterTextSplitter.
   Default chunk size is 500 characters with an 80-character overlap.

5. Each chunk is converted into a vector using the configured Ollama embedding
   model and stored in ChromaDB under the chroma_db directory.

6. When a question is submitted, the question is compared with the stored
   vectors. The default top four relevant chunks are retrieved.

7. The answer-generation model receives the question and retrieved context. Its
   prompt instructs it to use only that context, report when information is not
   available, and include source file citations.

8. The generated answer is evaluated with local checks. These checks calculate:
   - Relevance: the average Chroma similarity score for retrieved chunks.
   - Groundedness: the proportion of meaningful answer terms found in context.

9. If enabled, RAGAS evaluates the same question, answer, and retrieved context
   with an Ollama judge model. A supplied expected answer enables ground-truth
   context precision and context recall scoring.


Models and Services
-------------------

All model inference runs locally through Ollama at:

	http://localhost:11434

Current model responsibilities:

	Answer generation:   llama3.2
	RAGAS judge:          gemma3:4b
	Document embeddings:  mxbai-embed-large
	Vector database:      ChromaDB

The default embedding model produces 1024-dimensional vectors. The installed
environment may also contain nomic-embed-text and llama3.2-vision, but they are
not the current defaults.

The RAGAS judge model is separate from the answer-generation model because the
judge must follow structured JSON output instructions for RAGAS metrics.


RAGAS Metrics
-------------

RAGAS produces four metrics:

	Faithfulness
		Whether claims in the answer are supported by retrieved context.

	Answer Relevancy
		Whether the answer directly addresses the user's question.

	Context Precision
		Whether the most useful retrieved chunks are ranked highest.

	Context Recall
		Whether the retrieved context contains the information needed to answer.

The RAGAS metrics run sequentially because a local Ollama model can time out when
several requests compete for the same model instance. Each metric is isolated,
so a failure in one metric does not erase successful values from the others.

RAGAS uses JSON output mode because its parsers expect structured responses. If
the judge returns explanatory prose instead of the required schema, that metric
will be reported as unavailable with an error message.

RAGAS is optional. The main answer and local quality checks still work when the
judge is disabled or unavailable.


User Interface
--------------

The Streamlit UI provides:

	- Multiple document upload
	- Document index status
	- Embedding model and vector dimension information
	- Question input
	- Optional expected answer input
	- RAGAS enable/disable toggle
	- Judge model input
	- Generated answer display
	- Local relevance and groundedness scores
	- RAGAS metric values and errors
	- Retrieved source chunks with relevance scores


Important Files
---------------

	app/streamlit_app.py
		Streamlit entry point and user interface.

	src/ragpipeline/rag_pipeline.py
		End-to-end orchestration: retrieval, generation, local evaluation, and
		optional RAGAS evaluation.

	src/ragpipeline/retrieval.py
		Loads ChromaDB and retrieves the top relevant chunks.

	src/ragpipeline/generation.py
		Builds the Ollama answer-generation chain and prompt.

	src/embeddingpipeline/embedding.py
		Reads PDF and Excel files, resolves embedding configuration, and manages
		Chroma collection settings.

	src/embeddingpipeline/chunking.py
		Splits extracted documents into overlapping chunks.

	src/embeddingpipeline/vector_store.py
		Creates and loads the persistent Chroma vector store.

	src/evaluation/evaluation.py
		Implements local relevance and groundedness checks.

	src/evaluation/ragas_judge.py
		Configures the Ollama RAGAS judge, executes the four metrics, and reports
		per-metric scores and errors.

	src/evaluation/ragas_eval.py
		Command-line batch evaluation workflow for question/reference files.

	tests/evaluation_questions.json
		Sample questions and expected answers for evaluation.

	chroma_db/
		Persistent ChromaDB data. It can be reused between application runs.

	requirements.txt
		Python dependencies for Streamlit, LangChain, Ollama, ChromaDB, RAGAS,
		PDF processing, and Excel processing.


Configuration
-------------

Configuration is read from environment variables. The optional file
config/.env is currently empty and can be used as a place to document local
configuration, but the application does not automatically load it as a dotenv
file.

	OLLAMA_BASE_URL
		Ollama endpoint. Default: http://localhost:11434

	OLLAMA_LLM_MODEL
		Answer-generation model. Default: llama3.2

	OLLAMA_EMBED_MODEL
		Embedding model. Default: mxbai-embed-large

	RAGAS_JUDGE_MODEL
		RAGAS judge model. Default: gemma3:4b

	RAGAS_JUDGE_TIMEOUT
		Per-metric judge timeout in seconds. Default: 300

	RAGAS_JUDGE_ENABLED
		Enables or disables RAGAS. Values 0, false, no, and off disable it.

	CHROMA_PERSIST_DIRECTORY
		ChromaDB persistence directory. Default: chroma_db


Installation and Startup
------------------------

The project uses the virtual environment located at the parent project level:

	C:\Users\ganga.r.battem\AI_Projects\project1\.venv

From this project directory, start the application with:

	& "..\..\..\.venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py

The application is available at:

	http://localhost:8501

The system Python may not contain Streamlit. If the command

	python -m streamlit run app/streamlit_app.py

reports "No module named streamlit", use the virtual-environment command above.

Before using the application, make sure Ollama is running and the required models
are installed:

	ollama pull llama3.2
	ollama pull gemma3:4b
	ollama pull mxbai-embed-large


Sample Evaluation Data
----------------------

tests/evaluation_questions.json contains these reference pairs:

	Question: What is the price of the Bugatti Chiron in 2025?
	Reference: The Bugatti Chiron is priced at 3,000,000 USD in 2025.

	Question: How many Audi vehicles were sold in Q4 2025?
	Reference: Audi sold 12 vehicles in Q4 2025.

	Question: How many total vehicles were sold across all quarters in 2025?
	Reference: A total of 50 vehicles were sold across all quarters in 2025.

Enter the matching reference answer in the Streamlit Expected answer field for
ground-truth evaluation. Without a reference, RAGAS uses the generated answer as
a self-reference for context precision and recall, which is less strict.


Command-Line Evaluation
-----------------------

The batch evaluator can generate and score multiple questions. It accepts one or
more source files:

	& "..\..\..\.venv\Scripts\python.exe" src\evaluation\ragas_eval.py `
		--questions tests\evaluation_questions.json `
		--source path\to\source.pdf

Repeat --source for additional PDF or Excel files. The evaluator prints each
question's four RAGAS metrics followed by average scores.


Operational Highlights
----------------------

	- Local-first architecture: documents, embeddings, answers, and evaluation
	  requests remain on the local machine.
	- Persistent indexing: an existing Chroma collection is reused when present.
	- Format support: PDF pages and Excel sheets are both supported.
	- Source traceability: source names and page/sheet information are retained.
	- Quality visibility: local checks and RAGAS metrics are displayed separately.
	- Fault tolerance: RAGAS failure does not prevent answer generation.
	- Per-metric reporting: one failed RAGAS metric does not hide other scores.
	- Ground-truth support: expected answers enable stricter evaluation.
	- Simple deployment: the application runs as a local Streamlit process.


Known Limitations
-----------------

	- RAGAS quality depends heavily on the local judge model following the
	  required JSON schema.
	- Local Ollama inference can be slow, especially for four sequential RAGAS
	  metrics.
	- The local groundedness score is a token-overlap heuristic, not a semantic
	  entailment model.
	- Uploaded files are stored in temporary directories for the current process.
	- The application currently uses a single local Ollama endpoint.
	- Changing the embedding model or vector dimension may require rebuilding the
	  Chroma collection.


