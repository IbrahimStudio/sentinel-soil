import json 
from pipeline.worker import run_one_job
from dotenv import load_dotenv
load_dotenv("vm.env") 


payload=json.load(open('payload.json','r',encoding='utf-8'))

print(run_one_job(payload)) 
