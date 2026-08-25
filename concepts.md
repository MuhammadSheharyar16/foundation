# Concept Baseline

In this I explain important AI terms in simple language, give a supplier/procurement examples for each, and say whether each is deterministic, probabilistic, or contextual.

# LLM (Large Language Model)

An AI model trained on lots of text to understand and create human-like language.

**Example:**
  I give it contract delivery document and ask "Summarize the document delivery policy." 

**Label:** 
  Probabilistic

# Prompt

The instructions or question given to an AI model.

**Example:**
  I give it contract delivery document and ask "What is the expected delivery time in days?" 

**Label:** 
  Contextual

# System Instruction  

System instructions that define the AI’s role, behavior, and rules for the entire conversation.

**Example:**
  You are a procurement assistant. Only use the provided documents. If the answer is not found, say INSUFFICIENT_EVIDENCE. 

**Label:** 
  Contextual

# Token

A small piece of a word or text that AI reads and understands.

**Example:**
  Supplier Meridian Procurement deliver 8 orders.

**Label:** 
  Deterministic

# Context Window

The maximum amount of text an AI can read and handle at one time.

**Example:**
If you have 1,000 pages of supplier documents, split them into small chunks and give the AI only the relevant parts instead of the whole document at once.

**Label:** 
  Deterministic

# Temperature

A setting that controls how creative or random the AI’s answers are.
* Low temperature → more predictable
* High temperature → more variation

**Example:**
In this ask AI two times, "Summarize the supplier payment terms."

* Low temperature: Answers are usually very similar.
* High temperature: Answers may use different wording.

**Label:** 
  Probabilistic

# Embedding

In this text converts into numbers, so that text with similar meanings gets similar numbers.

**Example:**
* The supplier reported a vehicle engine problem.
* The vendor has an issue with the car motor.

Above two examples are different words but similar meanting.

**Label:** 
  Deterministic

# Cosine Similarity

In this measures how similar two embeddings are. A higher score means the meanings are more similar.

**Example:**
* The supplier reported a vehicle engine problem.
* The vendor has an issue with the car motor.
 Above first two examples are High similarity.

* The contract renewal date is next month.
 Above third one example is Low similarity.

**Label:** 
  Deterministic

# Chunk

A small, meaningful part of a large document that AI can easily read and understand.

**Example:**
C003 contains the renewal and termination sections from DOC-002 as one small, searchable piece.

**Label:** 
  Deterministic

# Overlap

A small amount of repeated text between chunks so important information is not lost between them.

**Example:**
If a delivery rule is split between C001 and C002, chunk overlap keeps the full rule together so the AI does not miss important information.

**Label:** 
  Deterministic

# Lexical Retrieval

In this finds relevant text by matching the same words or terms, not their meaning.

**Example:**
A search for "supplier" may find relevant chunks, but a search for "vendor" may find minimum results even if both words mean the same thing.

**Label:** 
  Deterministic

# Semantic Retrieval

In this finds relevant text by comparing meanings, even when different words are used.

**Example:**
A question using "vendor" can still find chunks using "supplier" because their meanings are similar.

**Label:** 
  Deterministic

  
# RAG (Retrieval-Augmented Generation)

In this finds relevant information first, then gives it to the AI so it can answer using the provided evidence.

**Example:**
For “What is the supplier delivery policy?”, RAG finds C001 and C002 and gives them to the AI to create an answer based on those chunks, not its memory.

**Label:** 
  Contextual

# Grounding

Making the AI answer only from the provided evidence, not from its general knowledge.

**Example:**
If the document does not provide a late payment interest rate, the AI should say "INSUFFICIENT_EVIDENCE" instead of guessing a percentage.

**Label:** 
  Contextual

# Hallucination

When AI makes up information that sounds correct but is not supported by the documents.

**Example:**
AI might say “typically 2% per month” even though the documents never mention this rate—this is a hallucination.

**Label:** 
  Probabilistic

# Provenance

A record showing where the AI’s information came from, such as the document and chunk used.

**Example:**
* **Answer:** Payment is due in 30 days.
* **Source:** DOC-003 → Chunk C004.

This lets someone check the original document to verify the answer.

**Label:** 
  Deterministic

# Citation

Its shows the exact source or chunk that supports the AI’s answer.

**Example:**
"The delivery window is five working days [C001]" shows the exact chunk that supports the answer.

**Label:** 
  Contextual


# Basic Flow

```text
Question
   ↓
Retrieve
   ↓
 Chunks
   ↓
 Prompt
   ↓
  LLM
   ↓
Validate
   ↓
 Answer
```