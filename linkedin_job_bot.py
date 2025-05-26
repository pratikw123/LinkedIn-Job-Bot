from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from sentence_transformers import SentenceTransformer, util
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import torch
import time
import csv
import pickle
import os
import re
from keybert import KeyBERT
import spacy

# ========= Load Resume =========
resume_text = """Pratik Wadaskar
    Navi Mumbai | pratikwadaskar2589@gmail.com | +91 9049731756 | linkedin.com/in/pratik-wadaskar
    Summary
    Full-Stack Developer with 2.5 years of experience specializing in backend development using Node.js, Express.js, and
    MongoDB, alongside strong frontend expertise in React.js. Skilled in building scalable microservices, RESTful APIs, and
    deploying applications on AWS using Docker. Familiar with Python backend development principles and SQL/NoSQL
    database management. Passionate about delivering robust, cloud-native, and efficient full-stack solutions with a focus on
    performance optimization and clean architecture.
    Skills
    Programming Languages: Python, JavaScript, TypeScript
    Backend Frameworks: Node.js, Express.js
    Frontend Technologies:React.js, Redux, HTML5, CSS3, Bootstrap, CoreUI
    Databases: MongoDB, MySQL, OracleDB
    Cloud & DevOp: AWS (basic), Docker (basic), Kubernetes (learning)
    Tools & Platforms: Git, GitHub, Postman, Cypress, WinSCP, PM2, VS Code
    Core Competencies: RESTful API Development, Microservices Architecture, Cloud Deployment, Performance Optimization,
    Testing and Debugging, Database Management, Prompt Engineering
    Experience
    Software Development Engineer, Jio Platform Limited – Navi Mumbai Dec 2022 – Present
    • Developed and optimized scalable backend APIs using Node.js and Express.js, improving system response times by 25%.
    • Built and managed microservices for large-scale applications with MongoDB and MySQL databases.
    • Integrated Docker-based containerization for deployment, enhancing application stability and portability.
    • Leveraged AWS services (EC2, S3) for cloud-based solutions and automated workflows.
    • Enhanced frontend functionalities using React.js and Redux, improving user experience and engagement by 30%.
    • Collaborated with cross-functional teams (Product, Design, QA) to deliver end-to-end solutions for telecom reporting
    systems.
    • Automated data processing pipelines, reducing manual intervention and boosting operational efficiency by 20%.
    Education
    Sant Gadge Baba Amravati University, BE in Information Technology Aug 2018 – Jul 2022
    GPA: 8.97/10
    Projects
    BCE In-house Product
    • Developed a full-stack web application using React.js, Node.js, Express.js, and OracleDB to automate internal workflows
    and reduce operational latency.
    BCE Data Search Tool
    • Engineered a React.js-based search platform integrated with RESTful APIs to enhance data retrieval accuracy by 25%.
    BCE Reference Data Loader
    • Built a web-based tool for automated data ingestion and validation from binary Excel files, ensuring reliable data
    operations for non-technical users.
    Certifications
    • Advanced React, Introduction to Databases for Back-End Development – Meta
    • Advance Your Node.js Skills – LinkedIn Learning
    • APIs and Web Services, Web Security, Data Structures - LinkedIn Learning"""


model = SentenceTransformer("all-MiniLM-L6-v2")
# model = SentenceTransformer("all-mpnet-base-v2")
kw_model = KeyBERT(model)


# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Canonical skills / techs / roles
CANONICAL_SKILLS = [
    "html", "css", "javascript", "typescript", "react", "angular", "vue.js",
    "node.js", "express", "mongodb", "mysql", "postgresql", "redux",
    "rest api", "graphql", "docker", "kubernetes", "aws", "azure", "gcp",
    "devops", "jest", "mocha", "cypress", "next.js", "tailwind", "sass", "php",
    "python", "java", "spring", "flask", "django", "ci/cd", "agile", "scrum",
    "frontend", "frontend developer", "responsive design", "figma",
    "state management", "component-based development", "single page application",
    "user interface", "ux", "design system"
]

canonical_embeddings = model.encode(CANONICAL_SKILLS, convert_to_tensor=True)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def chunk_text(text, max_words=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks


def extract_keywords(text, top_n=100):
    raw_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 4), stop_words='english', top_n=top_n)
    return [kw for kw, _ in raw_keywords]


def semantic_filter_keywords(keywords, threshold=0.6):
    filtered = set()
    for kw in keywords:
        kw_embedding = model.encode(kw, convert_to_tensor=True)
        sim_scores = util.cos_sim(kw_embedding, canonical_embeddings)[0]
        if sim_scores.max().item() > threshold:
            filtered.add(kw.lower())
    return filtered


def extract_semantic_keywords(text):
    clean = clean_text(text)
    raw_keywords = extract_keywords(clean)
    return semantic_filter_keywords(raw_keywords)


def fuzzy_match_keywords(set1, set2, threshold=0.7):
    matched = set()
    for kw1 in set1:
        for kw2 in set2:
            sim = util.cos_sim(model.encode(kw1), model.encode(kw2)).item()
            if sim > threshold:
                matched.add(kw1)
                break
    return matched


def extract_resume_keywords(resume_text):
    return extract_semantic_keywords(resume_text)


def extract_jd_keywords(jd_text):
    return extract_semantic_keywords(jd_text)


def normalize_keyword(kw, threshold=0.6):
    kw_vec = model.encode(kw, convert_to_tensor=True)
    sim_scores = util.cos_sim(kw_vec, canonical_embeddings)[0]
    best_idx = int(sim_scores.argmax())
    if sim_scores[best_idx].item() >= threshold:
        return CANONICAL_SKILLS[best_idx]
    return None

def extract_structured_keywords(text, threshold=0.6):
    doc = nlp(text)
    candidate_keywords = set()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if 1 <= len(phrase) <= 40:
            candidate_keywords.add(phrase)

    for token in doc:
        if token.pos_ in {"PROPN", "NOUN"} and not token.is_stop:
            candidate_keywords.add(token.text.lower())

    # Normalize to canonical
    normalized_keywords = set()
    for kw in candidate_keywords:
        norm = normalize_keyword(kw, threshold)
        if norm:
            normalized_keywords.add(norm)

    return normalized_keywords

# resume_keywords = extract_resume_keywords(resume_text)
resume_keywords = extract_structured_keywords(resume_text)
resume_chunks = chunk_text(resume_text)
resume_chunks = [clean_text(chunk) for chunk in resume_chunks]
resume_chunk_embs = model.encode(resume_chunks, convert_to_tensor=True)

# print("Resume Keywords:", resume_keywords)


# ========= Chrome Setup =========
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-webrtc")
options.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
options.add_argument("--disable-media-router")
options.add_argument("--disable-extensions")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)


# ========= Open LinkedIn =========
driver.get("https://www.linkedin.com")


# ========= Load Cookies if available =========
cookies_file = "linkedin_cookies.pkl"
if os.path.exists(cookies_file):
    try:
        cookies = pickle.load(open(cookies_file, "rb"))
        for cookie in cookies:
            driver.add_cookie(cookie)
        driver.get("https://www.linkedin.com/feed")
        wait.until(EC.url_contains("/feed"))
        print("✅ Logged in using saved cookies.")
    except Exception as e:
        print("❌ Failed to load cookies, logging in manually:", e)
        driver.get("https://www.linkedin.com/login")
else:
    # ========= Login to LinkedIn manually =========
    driver.get("https://www.linkedin.com/login")


# ========= Check if login is needed =========
if "login" in driver.current_url:
    # ========= Credentials =========
    LINKEDIN_EMAIL = "pratikwadaskar123@gmail.com"
    LINKEDIN_PASSWORD = "#Theboys@123"

    wait.until(EC.presence_of_element_located((By.ID, "username"))).send_keys(LINKEDIN_EMAIL)
    driver.find_element(By.ID, "password").send_keys(LINKEDIN_PASSWORD + Keys.RETURN)
    wait.until(EC.url_contains("/feed"))

    # ✅ Save cookies for future use
    pickle.dump(driver.get_cookies(), open(cookies_file, "wb"))
    print("✅ Logged in and cookies saved.")

time.sleep(5)


# ========= Apply Filter =========
try:
    job_title = "react"
    location = "India"

    # Build filtered LinkedIn job search URL
    search_url = (
        "https://www.linkedin.com/jobs/search/?"
        f"keywords={job_title.replace(' ', '%20')}"
        f"&location={location.replace(' ', '%20')}"
        "&f_TP=1"        # Date posted: Past 24 hours
        "&f_TPR=r86400"  # Time range in seconds (24 hours)
        "&f_E=2%2C3"     # Experience: Entry level (2) and Associate (3)
        "&f_JT=F"        # Job type: Full-time
        "&f_F=it%2Ceng"  # Job functions: IT, Engineering
        "&f_T=9%2C25201%2C25194%2C24%2C25170%2C3172%2C25169%2C266"  # Industries
        "&sortBy=DD"     # Sort by: Date Descending
    )

    driver.get(search_url)
    print("✅ Opened LinkedIn job search with filters applied via URL")

    time.sleep(5)  # Wait for jobs to load and page to stabilize

except Exception as e:
    print(f"❌ Failed to load filtered job search: {e}")
    driver.quit()
    exit()


# Wait for the job list container
# job_list_container = wait.until(
#     EC.presence_of_element_located((By.CSS_SELECTOR, "div.mOTgvPgoPApZSOpHXeJfmbXFINfwYzlcHQw"))
# )

wait = WebDriverWait(driver, 10)

job_urls = set()
page_num = 1

while True:
    print(f"📄 Scraping Page {page_num}")

    # Wait for job list container
    # job_list_container = wait.until(
    #     EC.presence_of_element_located((By.CSS_SELECTOR, "div.BTfbliYCBvyYpCFqwhjLaJweVNZJQQTwY"))
    # )

    # Wait for the outer list container
    scaffold_container = wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "scaffold-layout__list"))
    )

    # Get the direct inner scrollable container
    scrollable_div = scaffold_container.find_element(By.XPATH, "./div")

    # print("Scrollable container class:", scrollable_div.get_attribute("class"))

    # Wait for job cards to appear
    wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".job-card-container__link"))
    )


    # 2. Scroll to load jobs on current page
    scroll_step = 300
    total_scrolls = 100
    last_height = 0

    for _ in range(total_scrolls):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollTop + arguments[1];",
                              scrollable_div, scroll_step)
        time.sleep(0.4)

        job_cards = scrollable_div.find_elements(By.CSS_SELECTOR, ".job-card-container__link")
        for card in job_cards:
            href = card.get_attribute("href")
            if href and "/jobs/view/" in href:
                job_urls.add(href)

        current_height = driver.execute_script("return arguments[0].scrollTop", scrollable_div)
        if current_height == last_height:
            break
        last_height = current_height

    print(f"🔍 Found {len(job_urls)} total job URLs after page {page_num}")

    time.sleep(2)

    # Check for "Next" pagination button
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, "button.jobs-search-pagination__button--next")
        if not next_button.is_enabled():
            print("🚫 'Next' button disabled. Reached last page.")
            break

        # Click the next button
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(3)  # Wait for next page to load
        page_num += 1
    except:
        print("🚫 'Next' button not found. Reached last page.")
        break

print(f"🔍 Found {len(job_urls)} job URLs.")

time.sleep(3)  # Wait for content to settle

job_links = []
urls = list(job_urls)

# Step 3: For each job URL, open it and extract job description
for i, link in enumerate(urls):  # limit to first 3 to avoid rate limits ([:5]))
    driver.get(link)
    time.sleep(3)  # Let page load

    try:
        # Wait until the job description container is loaded
        jd_container = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.jobs-description__container"))
        )

        # Try clicking 'See more' if the button is present
        try:
            see_more_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'jobs-description__footer-button') and span[text()='See more']]"))
            )
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", see_more_button)
            driver.execute_script("arguments[0].click();", see_more_button)
            time.sleep(1)  # Allow description to expand
        except Exception as e:
            print("No 'See more' button found or clickable")

        # Get all text inside that article
        jd_text = clean_text(jd_container.text)

        # print(f"Job Description Extracted: {jd_text}")

        jd_keywords = extract_structured_keywords(jd_text)
        # print("jd_keywords:", sorted(jd_keywords))

        matched_keywords = fuzzy_match_keywords(resume_keywords, jd_keywords)
        tech_score = len(matched_keywords) / len(jd_keywords) if jd_keywords else 0

        # resume_chunks = chunk_text(resume_text)
        # resume_chunks = [clean_text(chunk) for chunk in resume_chunks]
        # resume_chunk_embs = model.encode(resume_chunks, convert_to_tensor=True)

        jd_chunks = chunk_text(jd_text)
        jd_chunks = [clean_text(chunk) for chunk in jd_chunks]
        jd_chunk_embs = model.encode(jd_chunks, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(resume_chunk_embs, jd_chunk_embs)[0]
        max_sim = similarities.max().item()

        # Final weighted score
        # final_score = (0.7 * max_sim) + (0.3 * tech_score)
        final_score = (0.4 * max_sim) + (0.6 * tech_score)

        # print(f"[{i}] MaxSim: {max_sim:.2f}, TechScore: {tech_score:.2f}, FinalScore: {final_score:.2f}")
        # print("Matched Keywords:", sorted(matched_keywords))
        # print(f"Final Score: {final_score:.2f}")

        if final_score >= 0.6:
            job_links.append(link)
            print(f"[{i+1}] ✅ Pass (score: {round(final_score, 3)})")
        else:
            print(f"[{i+1}] ❌ Skip (score: {round(final_score, 3)})")

    except Exception as e:
        print(f"[{i}] ⚠️ Error reading job description: {e}")


# ========= Save Matching Jobs =========
with open("matching_jobs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([[link] for link in job_links])

driver.quit()
print("✅ Done! Matching jobs saved to 'matching_jobs.csv'")