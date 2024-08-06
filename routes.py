from flask import Flask, render_template , session, request, jsonify
import csv
from jobspy import scrape_jobs
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
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

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Tiger Home Page')

def webscraper():
    jobs = scrape_jobs(
        site_name=["linkedin"],
        search_term="IT",
        job_type="internship",
        location="france",
        results_wanted=1000,
        hours_old=24,
        country_indeed='france',
        linkedin_fetch_description=True
    )

    jobs['is_remote'].fillna(0, inplace=True)
    print(f"Found {len(jobs)} jobs")

    selected_columns = ['id', 'location', 'title', 'job_url', 'is_remote', 'description']
    selected_jobs = jobs[selected_columns]
    selected_jobs = selected_jobs.dropna(subset=['description'])

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
    # Combine all relevant columns into a single string for each row
    combined_texts = new_jobs.apply(
        lambda row: ' '.join([str(row[col]) for col in new_jobs.columns]), axis=1
    ).tolist()

    # Splitting long rows into smaller chunks if necessary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = [chunk for text in combined_texts for chunk in text_splitter.split_text(text)]
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Initialize Pinecone
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    index_name = "internquest"
    index = pc.Index(index_name)
    
    # Embed texts and upsert to Pinecone
    vectors = embeddings.embed_documents(texts)
    ids = [f"{i}" for i in range(len(vectors))]

    index.upsert(vectors=zip(ids, vectors))

    print("Embedded and stored the new job rows in the vector database.")
    return None

def parse_search():
    # Set up parser
    parser = LlamaParse(
        result_type="markdown"  # "markdown" and "text" are available
    )

    documents = parser.load_data(glob.glob("uploads/*.pdf"))
    
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    index_name = "internquest"
    index = pc.Index(index_name)
    
    response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
    )
    
    
    x = response.data[0].embedding
    
    results = index.query(
        namespace="ns1",
        vector=x,
        top_k=5,
        include_values=False,
        include_metadata=True
    )

    print(results)
    
    
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
        for job in output:
            print(f"*"*70 + "\n" + job["jobTitle"]+" :  " + job["location"] + "\n" + "*"*70 )
        session['Jobs'] = output
        return render_template('index.html')            
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
    pdf_files = glob.glob(os.path.join(UPLOAD_FOLDER, '*.pdf'))
    
    if pdf_files:        
        parse_result = parse_search()
        return internships(parse_result)
    else:
        return jsonify({'message': 'No PDF files found'}), 400

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=webscraper, trigger="cron", hour=3, minute=36)
    scheduler.start()

    app.run(debug=True)