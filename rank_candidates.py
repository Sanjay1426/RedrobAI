import os
import json
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Target skills extracted from the Job Description
REQUIRED_SKILLS = [
    "embeddings",
    "retrieval systems",
    "ranking systems",
    "python",
    "vector databases",
    "production ml",
    "ndcg",
    "map",
    "mrr"
]

def load_dataset(filepath):
    """Loads candidates dataset from a JSONL file."""
    if not os.path.exists(filepath):
        logger.error(f"Dataset file not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading dataset from {filepath}...")
    candidates = []
    try:
        # Load using pandas directly
        df = pd.read_json(filepath, lines=True)
        logger.info(f"Successfully loaded {len(df)} candidates.")
        return df
    except Exception as e:
        logger.warning(f"Failed to load with pandas direct read: {e}. Falling back to manual line-by-line parsing.")
        # Manual fallback
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        candidates.append(json.loads(line))
                except json.JSONDecodeError as err:
                    logger.error(f"Error parsing line {line_num}: {err}")
        df = pd.DataFrame(candidates)
        logger.info(f"Successfully loaded {len(df)} candidates via manual parsing fallback.")
        return df

def read_job_description(filepath):
    """Reads the job description text file."""
    if not os.path.exists(filepath):
        logger.error(f"Job description file not found: {filepath}")
        raise FileNotFoundError(f"Job description file not found: {filepath}")
    
    logger.info(f"Reading job description from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()
    logger.info("Job description read successfully.")
    return jd_text

def prepare_candidate_texts(df):
    """Concatenates candidate profile fields to create a rich search text, handling nulls."""
    logger.info("Preparing candidate texts for embedding generation...")
    
    texts = []
    for _, row in df.iterrows():
        headline = str(row.get("headline", "")).strip()
        summary = str(row.get("summary", "")).strip()
        current_title = str(row.get("current_title", "")).strip()
        
        # Skills might be a list or a string
        skills_val = row.get("skills", "")
        if isinstance(skills_val, list):
            skills_str = ", ".join([str(s) for s in skills_val])
        else:
            skills_str = str(skills_val).strip()
            
        parts = [headline, summary, skills_str, current_title]
        # Filter out empty strings
        combined_text = " ".join([p for p in parts if p])
        texts.append(combined_text)
        
    df["candidate_text"] = texts
    return df

def compute_skill_match_score(skills_val):
    """
    Computes a normalized skill match score [0.0, 1.0] based on key skills from the JD.
    """
    if not skills_val:
        return 0.0, []
        
    # Standardize candidate skills list
    if isinstance(skills_val, list):
        cand_skills = [str(s).lower().strip() for s in skills_val]
    else:
        cand_skills = [s.strip() for s in str(skills_val).lower().split(",")]
        
    matched_skills = []
    for req_skill in REQUIRED_SKILLS:
        # Check if the required skill matches any of the candidate's skills directly
        # or if the required skill is a substring of any candidate skill
        if any(req_skill in cand_skill or cand_skill in req_skill for cand_skill in cand_skills):
            matched_skills.append(req_skill)
            
    score = len(matched_skills) / len(REQUIRED_SKILLS)
    return score, matched_skills

def compute_experience_score(exp):
    """
    Computes the experience score:
    - 1.0 if inside the target 5-9 years range
    - Soft linear discount for under-experienced: y / 5.0
    - Soft linear decay for over-experienced: 1.0 - (y - 9) * 0.1 (capped at 0.0)
    """
    try:
        y = float(exp)
    except (ValueError, TypeError):
        return 0.0
        
    if 5.0 <= y <= 9.0:
        return 1.0
    elif y < 5.0:
        return y / 5.0
    else:
        # Penalize overqualification slightly to target 5-9 range, but don't drop to 0 immediately
        return max(0.0, 1.0 - (y - 9.0) * 0.1)

def compute_behavioral_score(row):
    """
    Computes a weighted composite behavioral score [0.0, 1.0]:
    - Recruiter Response Rate: 30%
    - GitHub Commits (normalized to a cap of 200): 20%
    - Interview Completion Rate: 30%
    - Open to work: 20%
    """
    response_rate = float(row.get("recruiter_response_rate", 0.0))
    
    commits = float(row.get("github_commits_last_year", 0.0))
    normalized_commits = min(commits, 200.0) / 200.0
    
    completion_rate = float(row.get("interview_completion_rate", 0.0))
    
    open_to_work = 1.0 if bool(row.get("open_to_work", False)) else 0.0
    
    behavioral_score = (
        0.30 * response_rate +
        0.20 * normalized_commits +
        0.30 * completion_rate +
        0.20 * open_to_work
    )
    return behavioral_score

def generate_reasoning(row, sim_score, skill_score, matched_skills, exp_score, beh_score):
    """Generates detailed, high-quality reasoning based on all scores."""
    name = row.get("name", "Candidate")
    exp = row.get("years_of_experience", 0)
    commits = int(row.get("github_commits_last_year", 0))
    resp_rate = row.get("recruiter_response_rate", 0.0)
    open_status = "is actively open to work" if row.get("open_to_work") else "is passive/not open"
    
    # 1. Experience summary
    if exp_score == 1.0:
        exp_reason = f"perfect experience level of {exp} years (target: 5-9)"
    elif exp < 5:
        exp_reason = f"junior experience level of {exp} years (seeking 5-9)"
    else:
        exp_reason = f"highly senior experience level of {exp} years (exceeds target 5-9)"
        
    # 2. Skills summary
    if matched_skills:
        skills_reason = f"matching key skills: {', '.join(matched_skills)}"
    else:
        skills_reason = "lacking direct target skill matches"
        
    # 3. Behavioral summary
    if beh_score > 0.8:
        beh_reason = f"outstanding platform engagement (response rate: {resp_rate:.0%}, {commits} commits, {open_status})"
    elif beh_score > 0.5:
        beh_reason = f"moderate engagement (response rate: {resp_rate:.0%}, {commits} commits)"
    else:
        beh_reason = f"low platform engagement (response rate: {resp_rate:.0%}, {commits} commits)"
        
    reason = (
        f"{name} shows a strong semantic profile fit (similarity: {sim_score:.1%}) with a {exp_reason}. "
        f"They demonstrate strong technical capability, {skills_reason}. "
        f"Finally, they show {beh_reason}."
    )
    return reason

def run_ranking_pipeline():
    logger.info("Initializing Ranking Pipeline...")
    
    # Load dataset & job description
    df = load_dataset("candidates.jsonl")
    jd = read_job_description("job_description.txt")
    
    # Clean and concatenate text
    df = prepare_candidate_texts(df)
    
    # Generate embeddings
    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    logger.info("Encoding job description...")
    jd_embedding = model.encode(jd).tolist()
    
    logger.info("Encoding candidate texts...")
    candidate_texts = df["candidate_text"].tolist()
    candidate_embeddings = model.encode(candidate_texts, show_progress_bar=True).tolist()
    
    # Store in Qdrant In-Memory client
    logger.info("Initializing In-Memory Qdrant Client...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    client = QdrantClient(":memory:")
    
    collection_name = "candidates"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Insert candidate vectors
    logger.info("Upserting vectors into Qdrant...")
    points = []
    for idx, row in df.iterrows():
        points.append(
            PointStruct(
                id=idx,
                vector=candidate_embeddings[idx],
                payload={"candidate_id": row["candidate_id"]}
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    
    # Vector Search query
    logger.info("Performing vector search against Job Description...")
    search_response = client.query_points(
        collection_name=collection_name,
        query=jd_embedding,
        limit=len(df)
    )
    search_results = search_response.points
    
    # Build similarities map
    similarities = {res.id: res.score for res in search_results}
    df["semantic_similarity"] = df.index.map(similarities)
    
    # Calculate additional scoring dimensions
    logger.info("Calculating skill match, experience match, and behavioral scores...")
    
    skill_scores = []
    matched_skills_list = []
    for s in df["skills"]:
        score, matched = compute_skill_match_score(s)
        skill_scores.append(score)
        matched_skills_list.append(matched)
    df["skill_match"] = skill_scores
    df["matched_skills"] = matched_skills_list
    
    df["experience_match"] = df["years_of_experience"].apply(compute_experience_score)
    df["behavioral_score"] = df.apply(compute_behavioral_score, axis=1)
    
    # Compute hybrid ranking score
    logger.info("Computing final hybrid score (0.50 Semantic + 0.20 Skills + 0.15 Exp + 0.15 Behavior)...")
    df["final_score"] = (
        0.50 * df["semantic_similarity"] +
        0.20 * df["skill_match"] +
        0.15 * df["experience_match"] +
        0.15 * df["behavioral_score"]
    )
    
    # Generate text reasoning
    logger.info("Generating candidate-specific justifications...")
    reasonings = []
    for idx, row in df.iterrows():
        reason = generate_reasoning(
            row,
            row["semantic_similarity"],
            row["skill_match"],
            row["matched_skills"],
            row["experience_match"],
            row["behavioral_score"]
        )
        reasonings.append(reason)
    df["reasoning"] = reasonings
    
    # Sort and rank
    df_sorted = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1
    
    # Select columns for submission
    submission_df = df_sorted[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission_df.rename(columns={"final_score": "score"}, inplace=True)
    
    # Output to CSV
    output_csv = "team_redrob.csv"
    logger.info(f"Saving final rankings to {output_csv}...")
    submission_df.to_csv(output_csv, index=False)
    
    # Print results to stdout
    print("\n" + "="*80)
    print(f"{'RANK':<5} | {'ID':<10} | {'NAME':<15} | {'SIMILARITY':<10} | {'HYBRID SCORE':<12} | {'EXP (YRS)':<9}")
    print("="*80)
    for idx, row in df_sorted.iterrows():
        print(f"{row['rank']:<5} | {row['candidate_id']:<10} | {row['name']:<15} | {row['semantic_similarity']:<10.2%} | {row['final_score']:<12.4f} | {row['years_of_experience']:<9.1f}")
    print("="*80 + "\n")
    
    logger.info("Ranking Pipeline completed successfully!")

if __name__ == "__main__":
    run_ranking_pipeline()
