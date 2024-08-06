from flask import Flask, render_template
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
from flask import Flask, request, jsonify
import google.generativeai as genai
import json




load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model_internquest = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

# Ensure the directory to save files exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# if index_name not in pc.list_indexes():
#     print("Not Found")
# print(pc.list_indexes())

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
    except Exception as e:
        print(f"Error generating content: {e}")
        return None







internships("""
            3975414387,"linkedin","https://www.linkedin.com/jobs/view/3975414387","https://www.infineon.com/cms/en/careers/jobsearch/jobsearch/HRC0880258-Student-Job-Analog-IC-Design-f-m-div/#!source=400&urlHash=BFWh","Student Job: Analog IC Design (f/m/div)","Infineon Technologies","Le Puy-Sainte-Réparade, Provence-Alpes-Côte d'Azur, France","internship","2024-08-03","","","","","",0,"entry level","Education and Training","Semiconductor Manufacturing, Appliances, Electrical, and Electronics Manufacturing, and Computer Hardware Manufacturing","","","Are you looking for an opportunity to kick-off your career in an innovative company? Do you want to drive the development of new automotive solutions within an international environment? Then this is the opportunity that you are looking for: apply now and join our team in Provence! As an Analog/ Mixed-Signal working student, you will support our team in developing power-integrated circuits (ICs) for the automotive market. You will be part of a highly skilled team in constant interaction with other groups across R&D.
   

  

 In your new role you will:
   

  

* Work with Analog and Digital IC design industry experts;
* Document and present in detail the performance of your design;
* Support pre-silicon verification at block and/or top level according to the design flow;
* Support the review and analysis of the product's requirements;
* Be involved in IC debugging in the lab.


 You have a proactive personality and are a great communicator, willing to actively support and collaborate with your team members. You thrive in an international environment and are always looking for opportunities to develop your skills.
   

  

 You are best equipped for this task if you:
   

  

* Are a Master's degree Student in Electronics;
* Have team spirit, good communication and organizational skills;
* Are willing to work in an international environment;
* Are fluent in English.


 It would be an advantage if you have:
   

  

* Experience using Cadence tools for mixed-signal IC design and layout tools;
* Experience with Unix/Linux shell commands and editors;
* Experience with simulator (ELDO, Spectre, etc.);
* Experience with HDL (System Verilog, VHDL, etc.), python;
* Knowledge in Analog, Power electronics and Physics of semiconductor devices.


**Driving decarbonization and digitalization. Together.**
 Infineon designs, develops, manufactures, and markets a broad range of semiconductors and semiconductor-based solutions, focusing on key markets in the automotive, industrial, and consumer sectors. Its products range from standard components to special components for digital, analog, and mixed-signal applications to customer-specific solutions together with the appropriate software.
   

  

**– Automotive (ATV) shapes the future of mobility with micro-electronics enabling clean, safe and smart cars –**
 Semiconductors are essential to realize key trends like eMobility, automated driving and secure, connected cars. Infineon ATV is the #1 semiconductor partner in the fast-changing automotive world, based on our system knowledge coupled with our passion for innovation and quality. We are a key driver in the ever-advancing pace of digitalization in the automotive industry
   

  

**We are on a journey to create the best Infineon for everyone.**
 This means we embrace diversity and inclusion and welcome everyone for who they are. At Infineon, we offer a working environment characterized by trust, openness, respect and tolerance and are committed to give all applicants and employees equal opportunities. We base our recruiting decisions on the applicant´s experience and skills.
   

  

 We look forward to receiving your resume, even if you do not entirely meet all the requirements of the job posting.
   

  

 Please let your recruiter know if they need to pay special attention to something in order to enable your participation in the interview process.
   

  

 Click here for more information about Diversity & Inclusion at Infineon.","https://de.linkedin.com/company/infineon-technologies","","","","","","https://media.licdn.com/dms/image/C4E0BAQG0dfCs_tMDUQ/company-logo_100_100/0/1631307630627?e=2147483647&v=beta&t=kNPYrittKaUXRN5TPOqAFNtuc4EaLXcvnvoYW-JE2ww","","",""
3892557908,"","https://www.linkedin.com/jobs/view/3892557908","","Apprenticeship - Treasury Relative Grade - F/M","","Paris, Île-de-France, France","","","","","","","",0,"","","","","","**Job Description**
**Technip Energies - Together, pushing back the boundaries to shape a sustainable future**
 Technip Energies is a leading engineering and technology company serving the energy transition, with leading positions in Liquefied Natural Gas (LNG), hydrogen and ethylene, and a strong presence in the growth markets of blue and green hydrogen, sustainable chemistry, floating offshore wind turbines and CO2 capture and storage. The company benefits from its solid project delivery model, supported by an extensive offering of technologies, products and services.
   

  

 Backed by our commitments to Diversity & Inclusion and Sustainable Development (here), with a presence in 35 countries, our 15,000 employees are fully committed to bringing our customers' innovative projects to life, pushing back the boundaries of what is possible to accelerate the energy transition to a better future.
   

  

 Technip Energies is a leading engineering and technology company dedicated to the energy transition.
   

  

**Specifically, you will be required to :**
 Working within the Cash Management Paris Shared Services Center department in charge of operating entities covering mainly Europe, the Middle East, and other operating entities, and reporting to the center manager, the work-study student will have the following missions (non-limiting):
   

  

 Participate in the day-to-day cash management of the operating entities: daily reconciliation, management of cash requests and shipments to the cash center, position management, monitoring of cash outflows and inflows to bank accounts, ..., in the Diapason cash management software.
   

  

 Participate in the preparation of various reports and their improvement for distribution to operational entities:
   

  

* Weekly short term forecast reporting,
* Fortnightly Cash & Debt Report at the level achieved with Hyperion GTR,
* Quarterly Treasury Forecast Long Term (TREFLT) for one-year cash flow forecasts.


 Review of existing processes for potential improvement and simplification, drafting of operating procedures or training presentations for EMIA operating units.
   

  

 Participate in projects involving deployment of Diapason treasury software in countries/subsidiaries not using this system, ERP changeover with non-regression tests on data and reporting in the various treasury systems, development of automation and digitization, etc.
   

  

 During this apprenticeship, you will be accompanied by a tutor to answer any questions you may have.
   

  

**Profile:**
* Education : BAC+4/BAC+5
* School: Grande Ecole de Commerce/University (M2 level preferred) with finance/management specialization - specialized training in treasury would be a plus
* Skills required: rigorous, organized, team player, good communication skills, aptitude for using IT systems related to finance/treasury.
* Experience: previous experience in a cash management internship in a multinational company at least.
* Level of English: able to communicate with internal and external contacts in English and draft documents (e-mails/presentations/operating procedures).


 Are you looking to complete your apprenticeship in a dynamic, international environment, take part in large-scale projects and meet technical and organizational challenges?
   

  

**Then send us your application!**
 All our vacancies are open to people with disabilities.
   

  

**What's next?**
 Find out about all our vacancies on our career site:
   

  

 https://careers.hr.ten.com/
   

  

 Find out more about Technip Energies on Youtube :
   

  

 https://www.youtube.com/watch?v=8Mxj9h6vUjk
   

  

 Find out more about our ESG commitments on the following page:
   

  

 https://www.ten.com/sustainability","","","","","","","","","",""
3973476314,"","https://www.linkedin.com/jobs/view/3973476314","","Internship : Student lawyer / Labor law specialist - F/M","","Paris, Île-de-France, France","","","","","","","",0,"","","","","","**Job Description**
**Technip Energies - Together, pushing back the boundaries to shape a sustainable future**
 Technip Energies is a leading engineering and technology company serving the energy transition, with leading positions in Liquefied Natural Gas (LNG), hydrogen and ethylene, and a strong presence in the growth markets of blue and green hydrogen, sustainable chemistry, floating offshore wind turbines and CO2 capture and storage.
   

  

 The company benefits from its solid project delivery model, supported by an extensive offering of technologies, products and services.
   

  

 Backed by our commitments to Diversity & Inclusion and Sustainable Development (here), with a presence in 35 countries, our 15,000 employees are fully committed to bringing our customers' innovative projects to life, pushing back the boundaries of what is possible to accelerate the energy transition to a better future.
   

  

**Proposed tasks**
 Reporting to the Director of Employment Law within the Human Resources Department, you will apply your legal knowledge in managing individual and collective labor law relations on behalf of several legal entities in the Technip Energies group.
   

  

 You will mainly contribute to the following missions :
   

  

* Carry out research, benchmarking, legal studies and analyses on any individual or collective labor law subject;
* Prepare negotiation meetings, participate in the drafting of collective agreements, and monitor and report on agreements reached;
* Draft various legal documents: CSE information notes, employment contracts, correspondence, legal memos, internal communication articles, etc;
* Collaborate with the various HR contacts and internal departments (Compensation & Benefits,
* Recruitment, Adm/Payroll, etc.) to ensure the correct application of social standards and participate in the deployment of HR projects;
* Keep abreast of the latest social legislation, agreements and case law.


 Your tutor will support your integration and your development of skills. The diversity of the files will contribute to the trainee's intellectual enrichment and professional development.
   

  

**Profile Required**
* You are a student lawyer looking for a PPI internship and hold a Master 2 specialized in labor law, or are studying for a Master 2 specialized in labor law,
* Initial experience (internship, work-study program, etc.) gained in a company and/or law firm would be a plus.


**Skills Required**
* You have a very good grounding in employment law (individual and collective relations) and would like to put this to good use within a major international group;
* You have good analytical and summarizing skills, as well as strong writing skills;
* You are recognized for your rigor, organization and responsiveness, and you have excellent interpersonal skills;
* You are familiar with legal research sites and databases, Pack Office and IT tools;
* A good level of English is a plus.


 During this internship, you will be accompanied by a tutor to answer any questions you may have. This internship will give you the opportunity to develop your skills in a multicultural and dynamic environment.
   

  

**Send us your application !**
 All our vacancies are open to people with disabilities.
   

  

**What's next ?**
 You can find all our vacancies on our career site:
   

  

 https://careers.hr.technipenergies.com/
   

  

 Find out more about Technip Energies on Youtube: https://www.youtube.com/watch?v=8Mxj9h6vUjk
   

  

 Find out more about our ESG commitments on the following page:
   

  

 https://www.technipenergies.com/sustainability","","","","","","","","","",""
3991432084,"","https://www.linkedin.com/jobs/view/3991432084","","INTERNSHIP - Product Manager (R&D)","","Boulogne-Billancourt, Île-de-France, France","","","","","","","",0,"","","","","","We are looking for an apprentice to assist our Product Management team with the launch of our new product!
 



 As a Product Manager you are responsible for one of the products in our range. You work closely with our talented engineers, giving them the product vision, defining the development roadmap and priorities, testing the product and finally launching it into the IoT market.
 



 Among your missions, you will:
 


* help develop the mobile applications, and ensure that each screen is user friendly, clear and coherent.
* organize the beta tests and be a key actor in solving bugs.
* follow the industrialization process and the quality checks of the first batches.
* be responsible for the user experience and help put in place a flawless customer support
* collaborate with the Marketing team to ensure a successful product launch.



  





**Qualifications** 




  





 You have an engineering academic background, you are doing a business specialization and you are looking for an internship.
 



 You understand the technical concepts of software development, sensors, hardware development, logistics and supply chain. A complementary degree in Business (entrepreneurship, MBA, Marketing…) is a plus.
 


* You combine analytical and practical sense. You are “solution oriented”.
* You like (and manage!) to get results.
* You are fluent in English (oral & written) and in French
* You have a deep understanding of ergonomics and have a strong user empathy.
* You have a strong will to create the best possible user experience for our customers
* You are able to lead end-to-end projects.","","","","","","","","","",""
3884783261,"","https://www.linkedin.com/jobs/view/3884783261","","Product Data Scientist Intern","","Paris, Île-de-France, France","","","","","","","",0,"","","","","","**What You'll Do:**
 The Product Analytics & Data Science team uses cutting-edge technology, advanced statistics and machine learning to tackle some of the most complex Product challenges at Criteo. We help Criteo validate and evolve our products while exploring strategic game-changers to vault Criteo ahead in a fast-evolving media landscape. The team brings data and business expertise to feature teams, a unique understanding of complex Criteo machinery and supports Product Managers to design and build products with difference.
   

  

 Wondering how is the life in the Product Analytics & Data Science team?
   

  

 Take a peek at : https://careers.criteo.com/en/criteo-life-blog/from-the-inside/live-my-life-with-the-product-analytics-data-science-team/
   

  

 You will be assigned to one or several projects. The topics we tackle are wide and always evolving!
   

  

* Support the Measurement and R&D teams to build a comprehensive, practical, and universal measurement framework that frames the way CMOs understand their online marketing efficiency along the entire buyer journey
* Improve Identity & Privacy solutions by optimizing our capability to recognize users across all their devices and their interactions in the open-internet
* Work with the Quality Ad Experience team to pivot our user-level personalization engine and delivery rules to contextual and/or audience-based strategy
* Explore and support the development of our New Marketing Outcomes: Video & CTV, Contextual, and Omnichannel.
* Build the Buyer Index, a decision support service to improve the performance of digital ad campaigns in meeting clients’ goals across all addressability scenarios: addressable, cohort-based, or contextual.
* Work with the Trading Strategies team, to offer a suite of controls to our advertisers and enhance the performance of our business models: budget, audience, targeting.


 Overall, your responsibilities include:
   

  

* Mine large data sets and turn them into understandable and actionable insights
* Build scalable analytic solutions using state of the art tools based on large and granular datasets
* Design and execute a stream of analysis and tests to measure the impact of your solutions
* Master our internal analytic datasets and reporting tools


**Who You Are:**
* Master’s degree student or higher in a quantitative field (Mathematics, Computer Science, Physics, Engineering, Economics, etc.)
* Available for maximum 6 months, for end of study or gap year internship
* Outstanding analytical skills and creative thinking
* Fluency in the core toolkit of Data Science:
* Python; SQL/Hive/Presto
* Manipulating large-scale data sets
* Building data pipelines
* Descriptive and predictive modeling
* Implementing visualizations, dashboards, and reports
* Excellent interpersonal and communication skills, pro-active and independent to work with!


 We acknowledge that many candidates may not meet every single role requirement listed above. If your experience looks a little different from our requirements but you believe that you can still bring value to the role, we’d love to see your application!
   

  

**Who We Are:**
 Criteo is the global commerce media company that enables marketers and media owners to deliver richer consumer experiences and drive better commerce outcomes through its industry leading Commerce Media Platform.
   

  

 At Criteo, our culture is as unique as it is diverse. From our offices around the world or from home, our incredible team of 3,600 Criteos collaborates to develop an open and inclusive environment. We seek to ensure that all of our workers are treated equally, and we do not tolerate discrimination based on race, gender identity, gender, sexual orientation, color, national origin, religion, age, disability, political opinion, pregnancy, migrant status, ethnicity, marital or family status, or other protected characteristics at all stages of the employment lifecycle including how we attract and recruit, through promotions, pay decisions, benefits, career progression and development. We aim to ensure employment decisions and actions are based solely on business-related considerations and not on protected characteristics. As outlined in our Code of Business Conduct and Ethics, we strictly forbid any kind of discrimination, harassment, mistreatment or bullying towards colleagues, clients, suppliers, stakeholders, shareholders, or any visitors of Criteo. All of this supports us in our mission to power the world’s marketers with trusted and impactful advertising encouraging discovery, innovation and choice in an open internet.
   

  

**Why Join Us:**
 At Criteo, we take pride in being a caring culture and are committed to providing our employees with valuable benefits that support their physical, emotional and financial wellbeing, their interests and the important life events. We aim to create a place where people can grow and learn from each other while having a meaningful impact. We want to set you up for success in your job, and an important part of that includes comprehensive perks & benefits. Benefits may vary depending on the country where you work and the nature of your employment with Criteo. When determining compensation, we carefully consider a wide range of job-related factors, including experience, knowledge, skills, education, and location. These factors can cause your compensation to vary.","","","","","","","","","",""
3973473954,"","https://www.linkedin.com/jobs/view/3973473954","","Internship- Legal Corporate","","Paris, Île-de-France, France","","","","","","","",0,"","","","","","**About Us**
**JOB DESCRIPTION**
 At Technip Energies, we believe in a better tomorrow, and we believe we can make tomorrow better. With approximately 15,000 talented women and men, we are a global and leading engineering and technology company, with a clear vision to accelerate the energy transition. Designing and delivering added value energy solutions is what we do.
   

  

 If you share our determination to drive the transition to a low-carbon future, then this could be the job for you. We are currently seeking a legal intern, reporting directly to the legal corporate Vice President to join our legal team based in Nanterre, Paris.
   

  

**About The Job**
* If you are a law student, passionate about company law and corporate legal matters;
* If you are looking for an opportunity to gain hands-on experience and make meaningful contributions;
* If you are ambitious and curious and you feel at home in an environment in which there is close collaboration;
* You have a great sense of responsibility and strive for the best results with your team;


 We are thrilled to announce that that our Corporate Legal Department, made of 6 professionals, is seeking a dynamic trainee to join us.
   

  

 This is an exciting opportunity for a driven and proactive law student to gain exposure to a range of legal work, in company law and, more generally, in corporate legal matters, whilst developing the skills and knowledge needed to build a career in law in a highly results-oriented and performance-driven environment with a focus on excellence.
   

  

 Principal Duties Include
   

  

 Specific:
   

  

* Participate in providing assistance in connection with corporate finance transactions, tax restructurings and other transactions;
* Assist in providing legal support to ensure that the Technip Energies’ group of companies comply with all applicable legal and regulatory requirements and minimizing the Technip Energies’ group of companies exposure to regulatory risk;
* Assist in providing the legal support required by other corporate functions, including IT, procurement, Public Affairs, ESG and real estate matters;
* Stay current with legal/regulatory developments, best practices in the relevant areas of law and assist in updating or implementing policies and procedures in compliance therewith;


**About You**
 We’d love to hear from you if your profile meets the following essential requirements:
   

  

* Master’s degree in law with a preference for corporate & business law;
* A first training experience would be appreciated;
* Excellent verbal and written communication skills;
* Fluency in French and English;
* Extreme attention to detail;
* Ability to maintain the highest level of confidentiality and preserve the integrity of information and processes.


**Inclusion Standards**
 In our continuous journey to developing and building culture of inclusion, we adhere to four Inclusion Gold Standards. And you?
   

  

* We challenge our biases and embrace diversity of thought ;
* No one has all the knowledge and solutions, collectively we do ;
* We foster a caring environment where people are respected, comfortable to share and be heard ;
* We promote active listening for effective decision and action.


**What’s Next?**
 Starting Date: September 2024
   

  

 Once receiving your system application, Recruiting Team will screen and match your skills, experience, and potential team fit against the role requirements. We ask for your patience as the team completes the volume of applications with reasonable timeframe. Check your application progress periodically via personal account from created candidate profile during your application.
   

  

 We invite you to get to know more about our company by visiting www.technipenergies.com and follow us on LinkedIn , Instagram , Facebook , Twitter , Youtube for company updates.","","","","","","","","","",""
3991946996,"","https://www.linkedin.com/jobs/view/3991946996","","Référent Marketing & Communication","","Guyancourt, Île-de-France, France","","","","","","","",0,"","","","","","ASCENCIA BUSINESS SCHOOL est à la recherche pour l'un de ses partenaires un Référent Marketing & Communication .
   

  

 C'est une entreprise innovante spécialisée dans les solutions technologiques avancées et les services numériques. Dédiés à fournir des solutions personnalisées qui répondent aux besoins spécifiques des clients dans divers secteurs. Rejoindre cette société en alternance, c'est intégrer une équipe dynamique et passionnée, engagée dans l'excellence et l'innovation.
   

  

 Détails des missions confiées :
   

  

 Marketing Ventes et Communication :
   

  

* En charge de la gestion, la mise à jour (update produits, traduction) des catalogues et site internet
* Création de leaflet produits en collaboration avec les chefs produits
* Création des E-news
* Organisation de salons (réservation, aménagement stand, commande et gestion des demo kits...)
* Aide à communication sur Social Medias (LindedIn, Facebook,...)
* Aide au déploiement de la marque : évènements DD, OEM. Diffusion des infos, gestion goodies,..
* Mise à niveau des formations produits pour les DD et OEM (.ppt), organisation de webinaires
* Prospection tel et recherche de duplication de succès
* Etude de marché produits ou segments (applications)
* Relation Presse et support à l'élaboration de communiqués de presse


 Compétences clés requises :
   

  

 Marketing, communication (+outils IT liés à ces activités)
   

  

 Webtools
   

  

 Social Media
   

  

 Anglais 21820013-55584","","","","","","","","","",""
3892771177,"","https://www.linkedin.com/jobs/view/3892771177","","Innovation Lab Engineer (Internship)","","Suresnes, Île-de-France, France","","","","","","","",0,"","","","","","Ingenico is the global leader in payments acceptance solutions. As the trusted technology partner for merchants, banks, acquirers, ISVs, payment aggregators and fintech customers our world-class terminals, solutions and services enable the global ecosystem of payments acceptance. With 40 years of experience, innovation is integral to Ingenico’s approach and culture, inspiring our large and diverse community of experts who anticipate and help shape the evolution of commerce worldwide. At Ingenico, trust and sustainability are at the heart of everything we do.
   
            
            """)