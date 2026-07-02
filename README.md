# AI Candidate Ranking System

An intelligent recruitment solution that ranks job applicants based on how well their profiles match a given job description. The system uses semantic embeddings, vector search, and hybrid scoring to identify the most suitable candidates automatically.

---

## Overview

Hiring the right candidate can be time-consuming when hundreds of resumes need to be reviewed. This project simplifies the process by comparing candidate profiles with a job description using Artificial Intelligence and Machine Learning techniques.

The system evaluates resumes based on:
- Semantic similarity
- Technical skills
- Years of experience
- Candidate metadata
- Hybrid ranking score

Finally, the ranked results are exported to an Excel file for easy analysis.

---

## Features

- AI-based resume ranking
- Semantic search using Sentence Transformers
- Fast vector similarity search with Qdrant
- Hybrid scoring algorithm
- Candidate metadata filtering
- Automatic ranking generation
- Excel report generation
- Easy-to-use Python implementation

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Sentence Transformers | Text Embeddings |
| PyTorch | Deep Learning Backend |
| Qdrant | Vector Database |
| Scikit-learn | Similarity Calculations |
| OpenPyXL | Excel Export |

---

## Project Directory

```text
AI-Candidate-Ranking-System/
│
├── rank_candidates.py
├── sample_candidates.json
├── candidate_schema.json
├── sample_submission.csv
├── job_description.docx
├── requirements.txt
├── README.md
│
└── output/
    └── candidate_ranking.xlsx
```

---

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Candidate-Ranking-System.git
cd AI-Candidate-Ranking-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy sentence-transformers torch scikit-learn qdrant-client openpyxl tqdm
```

---

## How to Run

Execute the following command:

```bash
python rank_candidates.py
```

The application will:

1. Read candidate information
2. Load the job description
3. Generate embeddings
4. Calculate semantic similarity
5. Compute hybrid scores
6. Rank all candidates
7. Save results to Excel

---

## Sample Ranking Output

```text
======================================================================
Rank   Candidate ID   Name              Score     Experience
======================================================================
1      cand_001       Sophia Turner      0.91      9 Years
2      cand_005       Emma Richardson    0.86      7 Years
3      cand_004       Noah Mitchell      0.79      5 Years
4      cand_002       Liam Anderson      0.71      3 Years
5      cand_003       Olivia Carter      0.63      11 Years
======================================================================
```

---

## Output

After execution, the ranking report will be generated automatically.

```
output/
└── candidate_ranking.xlsx
```

---

## Future Improvements

- Resume upload through a web interface
- PDF and DOCX resume parsing
- AI-powered skill extraction
- Candidate recommendation engine
- ATS integration
- Explainable AI scoring
- Interactive dashboard using Streamlit
- REST API using Flask or FastAPI

---

## Requirements

- Python 3.10 or above
- Internet connection (for downloading transformer models)
- Qdrant Vector Database
- Required Python libraries

---

## Why This Project?

- Automates resume screening
- Reduces manual effort
- Improves hiring efficiency
- Provides objective candidate ranking
- Uses modern NLP and vector search techniques

---

## Author

### **Sanjay K**

**AI & Machine Learning Enthusiast**

- Python Developer
- Machine Learning Enthusiast
- NLP and Semantic Search Learner
- Interested in AI-powered Recruitment Systems

---

## License

This project is released under the **MIT License**.

Feel free to use, modify, and distribute this project for educational and personal purposes.
