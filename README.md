# LLM-Powered-Text-to-SQL-App-with-AI-Agents

This project builds a GenAI pipeline for Text-to-SQL generation, enabling users to query a relational database using plain English. It leverages fine-tuned LLaMA-3 (hosted on Hugging Face) and Mistral models trained on the Spider dataset, orchestrated with LangChain agents for SQL generation, clarification, and optimization.

The system integrates multiple AI agents:

-> Generation Agent (LLaMA-3) — initial SQL creation

-> Clarification Agent (Gemini via LangChain) — asks follow-up questions when queries are ambiguous

-> Optimization Agent (OpenAI GPT-3.5 via LangChain) — validates and repairs SQL before execution

A Flask backend with an HTML/CSS/JS frontend delivers an interactive interface, while the architecture demonstrates scalable deployment practices using Docker and CI/CD pipelines.

Highlights ->

Models: Fine-tuned LLaMA-3 (hosted on Hugging Face), Mistral (baseline comparison), Gemini (clarification), OpenAI GPT-3.5 (optimization)

Dataset: Spider (Text-to-SQL benchmark) + SQLite Formula-1 database

Frameworks: LangChain (agent orchestration), Flask (backend), HTML/CSS/JS (frontend)

Agents: Generation, Clarification, Optimization

Engineering: Dockerized workflows, CI/CD automation, Hugging Face Inference Endpoint hosting
