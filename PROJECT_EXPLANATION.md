# Temporal Grounding & Verification Framework
**Complete Project Explanation & Codebase Guide**

This document explains exactly what we are building, how it works step-by-step, and provides a minute-to-minute breakdown of every file in the project so the team understands exactly how everything connects together.

---

## 1. What We Are Building
Large Language Models (LLMs) hallucinate, especially when it comes to time. They get dates wrong, mix up the order of events, or invent relationships that never happened. 

While Retrieval-Augmented Generation (RAG) helps by giving the LLM context, standard RAG doesn't *verify* if the final answer the LLM spits out is actually correct.

**We built a framework that fixes this.** 

Our framework:
1. **Retrieves** facts from a strict Temporal Knowledge Graph.
2. **Grounds** the LLM by forcing it to answer using *only* those facts.
3. **Extracts** every individual claim the LLM just made.
4. **Verifies** every single claim against the knowledge graph *after* generation.
5. **Explains** exactly why a claim is true, false, or impossible.

We proved this works by generating a **Temporal Hallucination Benchmark** of 20,050 facts (some true, some deliberately corrupted). Our verification engine caught the corrupted facts with **99.8% accuracy**.

---

## 2. Step-by-Step Workflow (The Pipeline)

When a user asks a question (e.g., *"Did the US threaten Russia in March 2014?"*), here is the exact sequence of events:

### Step 1: Query Parsing
We use Natural Language Processing (spaCy) to read the question. We extract the entities ("United States", "Russia") and the timeframe ("March 2014").

### Step 2: Knowledge Graph Retrieval
We take those entities and dates and query our **In-Memory Knowledge Graph** (which contains ~90,000 real-world geopolitical facts from 2014). We pull out only the facts that match the entities and the timeframe.

### Step 3: LLM Reasoning
We send a prompt to our local LLM (running via LM Studio). We give it the user's question AND the facts we just retrieved. We instruct it: *"Answer the question using ONLY these facts."*

### Step 4: Claim Extraction
The LLM generates a text answer. We then ask the LLM (or use NLP rules) to break its own answer down into strict logical claims in the format: `[Subject, Predicate, Object, Timestamp]`.
*Example extracted claim: `["United States", "Threaten", "Russia", "2014-03-03"]`*

### Step 5: Verification
This is the core of our project. We take that extracted claim and run it through a 4-check verification engine against our Knowledge Graph:
1. **Entity Check:** Does "United States" exist in the graph?
2. **Relation Check:** Did they ever "Threaten" Russia?
3. **Timestamp Check:** Did it happen exactly on "2014-03-03"?
4. **Ordering Check:** If multiple events are mentioned, are they in the right chronological order?

### Step 6: Classification & Explanation
Based on the verification, we classify the LLM's answer into one of 5 categories:
- **Supported:** 100% correct.
- **Partially Supported:** The event happened, but the date is slightly off.
- **Unsupported:** Total hallucination. It never happened.
- **Temporally Impossible:** The order of events contradicts known history.
- **Cannot Verify:** We don't have enough data in our graph to know.

We then generate a human-readable explanation showing exactly the evidence we found.

---

## 3. Minute-to-Minute Details: Every File Explained

The project is highly modular. Here is exactly what every file does and how it connects.

### Core Orchestration
- **`run_all.py`**: The master script. If you run this, it executes the entire pipeline end-to-end: downloads data, builds the graph, generates the benchmark, runs verification, and tests the LLM. It's the best place to start reading the code.
- **`download_data.py`**: A robust downloader. It tries to get the ICEWS14 dataset from HuggingFace, GitHub mirrors, and if all else fails, it generates a perfect synthetic dataset so the pipeline never breaks.
- **`config/settings.py`**: Central configuration. Contains all paths, API URLs (like LM Studio at port 1234), and threshold values (like how many days off a date can be to still be considered "Partially Supported").

### The Data Layer
- **`src/data_loader/icews_loader.py`**: Parses the raw ICEWS14 `.tsv` files. It converts the IDs into readable text names and converts strings into proper Python datetime objects.
- **`src/data_loader/cronquestions_loader.py`**: Parses a different dataset (CronQuestions) used for evaluating question-answering accuracy.

### The Knowledge Graph Layer
- **`src/knowledge_graph/memory_store.py`**: Our custom In-Memory Temporal Knowledge Graph using NetworkX. This is what makes the project run without needing to install Neo4j. It holds the entities, relations, and timestamps, and provides the `verify_fact()` function.
- **`src/knowledge_graph/graph_store.py` & `schema.py`**: The Neo4j equivalents. We built these for scalability if we ever want to move off the in-memory graph to a dedicated database.
- **`src/knowledge_graph/graph_builder.py`**: Orchestrates taking the loaded data (from `icews_loader.py`) and pushing it into the graph store.

### The Retrieval Layer
- **`src/retrieval/query_parser.py`**: The NLP engine that extracts entities and normalises dates (e.g., turning "March 2014" into a start/end date range).
- **`src/retrieval/query_templates.py`**: The actual query structures used to pull facts out of the graph (e.g., "Find all events between Entity A and Entity B").
- **`src/retrieval/temporal_retriever.py`**: Takes the parsed query, runs the template against the graph, and scores the relevance of the returned facts.

### The Reasoning Layer
- **`src/reasoning/llm_reasoner.py`**: Connects to LM Studio via the `openai` python package. It handles sending the prompt and evidence to the local Qwen3 model and getting the response.
- **`src/reasoning/prompt_templates.py`**: Holds the exact text prompts we send to the LLM (instructing it to be grounded and use evidence).
- **`src/reasoning/claim_extractor.py`**: Breaks the LLM's paragraph response into `[Subject, Predicate, Object, Timestamp]` claims using NLP rules or a secondary LLM call.

### The Verification Layer (The Engine)
- **`src/verification/verification_result.py`**: The data models. Defines what a `TemporalClaim`, `TemporalFact`, and `VerificationResult` look like in code.
- **`src/verification/temporal_verifier.py`**: The 4-stage checker. It takes an extracted claim and runs the Entity, Relation, Timestamp, and Ordering checks against the graph store.
- **`src/verification/consistency_checker.py`**: Advanced validation that ensures multiple facts don't logically contradict each other chronologically.

### Detection, Explanation & Hallucination
- **`src/hallucination/detector.py`**: Takes the results from the verifier and assigns the final label (Supported, Unsupported, etc.) and calculates the Temporal Consistency Score (TCS).
- **`src/explanation/explainer.py`**: Turns the raw code output of the detector into the nice English explanations the user sees.
- **`src/hallucination/corruption_strategies.py`**: When we generate our benchmark, this file contains the functions that deliberately corrupt true facts (e.g., swapping a date, changing an entity) to test if our verifier can catch them.
- **`src/hallucination/benchmark_generator.py`**: Uses the corruption strategies to build the 20,050 example benchmark.

### Evaluation
- **`evaluation/metrics.py`**: Calculates the math: Accuracy, Precision, Recall, and F1 Score.
- **`evaluation/baselines.py`**: Defines "Vanilla LLM" and "Standard RAG" setups so we have something to compare our system against to prove ours is better.
- **`evaluation/run_evaluation.py`**: The script that actually runs the experiments and spits out the final metrics.
- **`src/pipeline/pipeline.py`**: The glue that ties Steps 1 through 6 together into a single, callable `.process()` function.

---

## 4. Why This Matters

We built this entirely on free, open-source technology. 
- We don't pay for OpenAI because we use **LM Studio**.
- We don't pay for database hosting because we built an **In-Memory Graph**.
- We solved a real research problem: LLMs are terrible at time. We built a mathematical, verifiable way to stop them from hallucinating history.
