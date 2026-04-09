# Semantic Matcher: 2025 Judicial Elections Data Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML/CSS](https://img.shields.io/badge/HTML5_&_CSS3-E34F26?style=for-the-badge&logo=html5&logoColor=white)

## Context & Problem Statement
In 2025, Mexico held its first-ever popular election for federal judges and Supreme Court ministers. The National Electoral Institute (INE) published an open dataset containing the profiles, academic backgrounds, and narrative essays of **3,414 candidates**. 

**The Challenge:** While standard data analysis handles structured numbers, civic data is overwhelmingly unstructured text. Traditional keyword searches fail because a voter looking for "social justice" won't find a candidate who wrote "equity and human rights" unless the system actually understands context.

**The Solution:** I built an end-to-end **Natural Language Processing (NLP)** pipeline and a zero-latency static web application to act as a **Semantic Matcher**, connecting voters with candidates based on the actual meaning of their proposals.

---

## Technical Architecture 

This project is divided into two main layers: The **Backend ML Pipeline** and the **Frontend Delivery Engine**.

### 1. Data Pipeline & NLP (Python)
Before deploying the app, the raw data underwent rigorous mathematical processing:
* **Data Cleaning:** Identified and filtered out anomalous records (cancelled or incomplete applications) to ensure the ML model only trained on the 3,369 officially published candidates.
* **Feature Engineering (Credential Scoring):** Developed a weighted algorithm to assign an "Academic Score" to candidates based on their legal background.
* **Natural Language Processing (NLP):** Combined candidates' visions and proposals into a single text corpus. Applied **TF-IDF Vectorization** using *N-grams (2,3)* and custom domain-specific stopwords to isolate meaningful political rhetoric.
* **Topic Modeling (LDA):** Deployed **Latent Dirichlet Allocation (LDA)** to mathematically categorize candidates into 5 "Narrative Pillars" (e.g., *Social Defense, Digital Efficiency, Constitutional Formalist*).
* **Data Serialization:** Exported the processed multidimensional data into a lightweight JSON file for deployment.

### 2. Frontend & Search Engine (Vanilla JS)
* **Static Site Architecture:** Designed a serverless, static web app that fetches the pre-calculated ML JSON upon loading, enabling fast, zero-cost deployment via GitHub Pages.
* **Semantic Search Algorithm:** Built a client-side search engine that matches user queries against the candidates' AI-generated Narrative Pillars, essays, and proposals—not just their names.
* **DOM Management:** Implemented custom algorithmic pagination and dynamic DOM injection to handle thousands of records without crashing the browser.
* **Data Visualization:** Integrated `Chart.js` for dynamic, real-time demographic charting.

---

## Live Demo
**https://csv-seb.github.io/mexico-judicial-2025/**

---

## Business / Industry Translation
While this project focuses on civic tech, the underlying architecture directly translates to high-value industry applications:
* **Retail / E-commerce:** The same NLP and Topic Modeling engine can group thousands of customer reviews to discover product flaws or cluster semantic search queries for better product matching.
* **Financial Services:** The static-search architecture can filter massive catalogs of financial products or parse through thousands of regulatory text files in milliseconds.

Author: **Sebastian Méndez Ramírez**
