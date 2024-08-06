from flask import Flask, render_template , session, request, jsonify
import csv
from jobspy import scrape_jobs
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_together.embeddings import TogetherEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from llama_parse import LlamaParse
import openai
import glob
import google.generativeai as genai
import json

from openai import OpenAI





load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model_internquest = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
client = OpenAI(api_key=OPENAI_API_KEY)
# Ensure the directory to save files exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'akjezrghualezhrulaehr'
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Tiger Home Page')

def webscraper():
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
        search_term="IT",
        job_type="internship",
        results_wanted=1000,
        hours_old=672
    )

    jobs['is_remote'].fillna(0, inplace=True)
    print(f"Found {len(jobs)} jobs")

    selected_columns = ['id', 'location', 'title', 'job_url', 'is_remote', 'description']
    selected_jobs = jobs[selected_columns]
    selected_jobs = selected_jobs.dropna(subset=['description','job_url'])

    file_path = "csv_Files/jobs.csv"

    if os.path.exists(file_path):
        existing_jobs = pd.read_csv(file_path)
        # Find new jobs that are not duplicates
        new_jobs = selected_jobs[~selected_jobs['id'].isin(existing_jobs['id'])]
        updated_jobs = pd.concat([existing_jobs, new_jobs])
        updated_jobs.drop_duplicates(subset=['id'], keep='last', inplace=True)
    else:
        new_jobs = selected_jobs
        updated_jobs = selected_jobs

    # Save updated jobs to CSV
    updated_jobs.to_csv(file_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

    print(f"Appended {len(new_jobs)} new jobs to csv file.")

    if not new_jobs.empty:
        # Embed and store new jobs in your vector database
        embed_and_store(new_jobs)

    return None

def embed_and_store(new_jobs):
    # Replace NaN with an empty string or some default value
    new_jobs.fillna('', inplace=True)

    combined_texts = new_jobs.apply(
        lambda row: ' '.join([str(row[col]) for col in new_jobs.columns]), axis=1
    ).tolist()
    print(combined_texts[:5])  # Debugging print statement

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    texts = [chunk for text in combined_texts for chunk in text_splitter.split_text(text)]
    print(texts[:5])  # Debugging print statement

    embeddings = TogetherEmbeddings(api_key="11ceeb85849e5e4bca6b2585107fbbeceff5af2d93792b5f10d93984379002e1", model="togethercomputer/m2-bert-80M-8k-retrieval")
    try:
        vectors = embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Error embedding documents: {e}")
        return  # Exit the function if there's an error

    print(vectors[:5])  # Debugging print statement

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "interquest1"
    index = pc.Index(index_name)

    # Prepare the IDs and metadata
    ids = new_jobs['id'].astype(str).tolist()
    job_metadata = new_jobs.to_dict(orient='records')
    print(job_metadata[:5])  # Debugging print statement

    # Prepare the vectors with IDs and metadata
    vectors_with_metadata = [
        {"id": id_, "values": vector, "metadata": metadata} 
        for id_, vector, metadata in zip(ids, vectors, job_metadata)
    ]

    # Validate the vectors_with_metadata structure
    for item in vectors_with_metadata[:5]:
        print(item)  # Debugging print statement

    try:
        # Upsert the vectors into Pinecone
        index.upsert(vectors=vectors_with_metadata, namespace="ns1")
        print("Embedded and stored the new job rows in the vector database.")
    except Exception as e:
        print(f"Error during upsert: {e}")



def parse_search():
    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(glob.glob("uploads/*.pdf"))
    document = documents[0].text

    messages = f"""
    You are a CV filter system. Use these informations in the CV: '{document}'. Note that the output should be in JSON format:
    {{
        "Informations": [
            {{
                "job_to_search_for": str,
                "Work Experience": str (example: 2 years),
                "Key_Responsibilities_and_Achievements": [str, str, str, ...],
                "Skills": [str, str, str, str, ...],
                "Certifications": [str, str, str, ...],
                "Projects": [str, str, str, ...],
                "recap": str
            }}
        ]
    }}
    """
    
    response = model_internquest.generate_content(
        messages,
        generation_config=genai.GenerationConfig(temperature=0)
    )
    print("\n" + "*"*60 + " Internships : " + response.text + "*"*60 + "\n")
    json_response = json.loads(response.text)
    info = json_response["Informations"][0]

    job_to_search_for = info["job_to_search_for"]
    work_experience = info["Work Experience"]
    responsibilities = ", ".join(info["Key_Responsibilities_and_Achievements"])
    skills = ", ".join(info["Skills"])
    certifications = ", ".join(info["Certifications"])
    projects = ", ".join(info["Projects"])
    recap = info["recap"]

    paragraph = (
        f"{recap} With {work_experience} of work experience im searshing for an internship in {job_to_search_for}, I have "
        f"been responsible for {responsibilities}. My skills include {skills}. "
        f"I have earned certifications such as {certifications}, and I have worked on projects like {projects}."
    )
    print(paragraph)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "interquest1"
    index = pc.Index(index_name)

    embeddings = TogetherEmbeddings(api_key="11ceeb85849e5e4bca6b2585107fbbeceff5af2d93792b5f10d93984379002e1", model="togethercomputer/m2-bert-80M-8k-retrieval")
    vector = embeddings.embed_query(paragraph)
    print(vector)  # Debugging print statement

    results = index.query(
    namespace="ns1",
    vector=vector,
    top_k=5,
    include_values=False,
    include_metadata=True
    )
    descriptions = [match['metadata']['description'] for match in results['matches']]
    combined_description = ' '.join(descriptions)
    combined_description = combined_description.replace('\n', ' ').replace('  ', ' ').strip()
    return combined_description
    
    
def internships(parse_search_result):
    messages = """
    You are an internship recommandation system.use these search results : """ + parse_search_result + """'. Note that the output should be in JSON format:
    {{
                    "jobs": [
                        {{
                            "jobTitle": str,
                            "link": str,
                            "description": str,
                            "location" : str
                        }},
                        {{
                            "jobTitle": str,
                            "link": str,
                            "description": str,
                            "location" : str
                        }},
                        {{
                            "jobTitle": str,
                            "link": str,
                            "description": str,
                            "location" : str
                        }},
                        {{
                            "jobTitle": str,
                            "link": str,
                            "description": str,
                            "location" : str
                        }},
                        {{
                            "jobTitle": str,
                            "link": str,
                            "description": str,
                            "location" : str
                        }}
                    ]
        
    }}
    """
    try:        
        response = model_internquest.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                temperature=0
            )
        )
        print("\n" + "*"*60 + " Internships : " + response.text + "*"*60 + "\n")
        json_response = json.loads(response.text)
        
        output = json_response.get("jobs", [])     
        print(output)   
        return output            
    except Exception as e:
        print(f"Error generating content: {e}")
        return None
    
@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return render_template('index.html', filename=file.filename)
    else:
        return jsonify({'message': 'Invalid file format. Only PDFs are allowed.'}), 400



@app.route('/get_internships', methods=['POST'])
def get_internships():  
    parse_result = parse_search()
    output = internships(parse_result)
    session['Jobs'] = output
    return(render_template("index.html", jobs=output))

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=webscraper, trigger="cron", hour=9, minute=54)
    scheduler.start()
    app.run(debug=True)