from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from langchain_community.llms import LlamaCpp
from langchain_experimental.agents import create_csv_agent
from fastapi.templating import Jinja2Templates #frontend library
from fastapi.staticfiles import StaticFiles #images
from fastapi import Request
import pandas as pd
import logging
from pydantic import BaseModel
import json


model_path="./llms/mistral-7b-v0.1.Q2_K.gguf"
csv_uploaded_path="csv/uploaded/"

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5173/"
]


logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = LlamaCpp(model_path=model_path, n_gpu_layers=-1, temperature=0, max_tokens=4096, n_ctx=4096)
    app.state.csv_agent = create_csv_agent(llm, './csv/MedGPT.csv', verbose=True)
    yield

app = FastAPI(
    title="Eecute model on MedGPT csv",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/execute", tags=["Execute"])
def _execute_model(request: Request, query: str):
    result = request.app.state.csv_agent.run(query)
    return result



templates = Jinja2Templates(directory="templates")

#images
app.mount("/static", StaticFiles(directory="static"), name='static')


# WEB INTERFACE ----------------------------------------------------------------------------------------

""" @app.get("/home")
def home(request: Request):
    left = 'Fernando'
    right = 'Nicotera'
    return templates.TemplateResponse(request, name="home.html", context={'id1': left, 'id2':right})
 """

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, name="provahome/index.html")

@app.get("/progressbar")
def progressbar(request: Request):

    return templates.TemplateResponse(request, name="progressbar.html")

@app.get("/testbootstrap")
def testbootstrap(request: Request):
    return templates.TemplateResponse(request, name="testbootstrap.html")

@app.get("/csv")
def testbootstrap(request: Request):
    return templates.TemplateResponse(request, name="csv.html")

@app.get("/Step3")
def Step3(request: Request):
    return templates.TemplateResponse(request, name="Step3.html")





""" @app.post("/senddata")
async def senddata(request: Request):
    body = request.stream()
    res = [i async for i in body]
    df = pd.json_normalize(body)
    df.to_csv('test.csv', index=False, encoding='utf-8')
 """

class ObjectListItem(BaseModel):
    item: str

@app.post("/senddata")
async def get_body(request: Request):
    data = await request.json()
    df = pd.json_normalize(data)
    print(df['table'])
    print(df['indvariables'][0][0])

    #n_files = len(os.listdir(csv_uploaded_path)) # your directory path

    #df.to_csv('csv/uploaded/csv_'+str(n_files)+'.csv', index=False, encoding='utf-8')