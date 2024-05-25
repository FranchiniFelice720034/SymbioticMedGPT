from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, status
from fastapi.responses import JSONResponse
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import Tool
from fastapi.templating import Jinja2Templates #frontend library
from fastapi.staticfiles import StaticFiles #images
from fastapi import Request
import pandas as pd
import logging
from pydantic import BaseModel
import os
import datetime
import time
import json 
import random

model_path="./llms/mistral-7b-v0.1.Q2_K.gguf"
csv_uploaded_path="csv/uploaded/"

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5173/"
]

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:

Last few messages between you and user, :
{chat_history_buffer}
"""

chat_history_buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history_buffer",
    input_key="input"
)

def perform_classification_fn(input:str) -> list[str]:
    columns = app.state.df.columns.tolist()
    random_columns = random.sample(columns, k=5)
    return random_columns
custom_classification_tool = Tool.from_function(
    func=perform_classification_fn,
    name="Classification with a dependent variable",
    description="Useful for when you are asked to perform a classification task on a pandas dataframe with \
        a dependent variable, you need to give to the tool in input the name of the dependent variable and \
        the tool will return the name of the firsts 5 most useful features to perform the classification.",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LlamaCpp(model_path=model_path, n_gpu_layers=-1, temperature=0, max_tokens=4096, n_ctx=4096)
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

@app.get("/execute", tags=["Use Model"])
def _execute_model(request: Request, query: str):
    if not hasattr(request.app.state, 'csv_agent'):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded")

    result = request.app.state.csv_agent.invoke(query)
    return result['output']

@app.post("/start-model-and-get-first-review", tags=["Use Model"])
async def _start_model_get_first_review(file: UploadFile = File(...), dep_var: str = Form(...)):

    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File required")
    if not dep_var:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="dep_var required")
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File format is not csv")
    
    df = pd.read_csv(file.file)

    if dep_var not in df.columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dep_var value")
    
    app.state.df = df
    app.state.dep_var = dep_var


    app.state.csv_agent = create_pandas_dataframe_agent(
        app.state.llm, df, include_df_in_prompt = True, 
        prefix=PREFIX, verbose=True,
        extra_tools=[custom_classification_tool],
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "input_variables":['df_head', 'input', 'agent_scratchpad', 'chat_history_buffer'],
            "memory": chat_history_buffer
        }
    )
    result = app.state.csv_agent.invoke(f'Perform a classification on a pandas dataframe with a \
                                        dependent variable, the dependent variable is called {dep_var}. \
                                        After doing the classification write me an argued reply where \
                                        you say that the features you found through your classification \
                                        tool as most important are the 5 that the tool gave you as output')
    return result['output']

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

@app.get("/", tags=["Frontend"])
def index(request: Request):
    return templates.TemplateResponse(request, name="provahome/index.html")

@app.get("/progressbar", tags=["Frontend"])
def progressbar(request: Request):

    return templates.TemplateResponse(request, name="progressbar.html")

@app.get("/testbootstrap", tags=["Frontend"])
def testbootstrap(request: Request):
    return templates.TemplateResponse(request, name="testbootstrap.html")

@app.get("/csv", tags=["Frontend"])
def testbootstrap(request: Request):
    return templates.TemplateResponse(request, name="csv.html")

@app.get("/Step3", tags=["Frontend"])
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

@app.post("/senddata", tags=["Frontend"])
async def get_body(request: Request):
    data = await request.json()
    df = pd.json_normalize(data['table'])
    header = df.loc[0, :].values.tolist()
    df = df.iloc[1: ]
    
    df.columns = header

    if(not os.path.isdir(csv_uploaded_path)):
        os.mkdir(csv_uploaded_path)
    
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y-%H-%M-%S')
    print(df)
    df.to_csv('csv/uploaded/csv_'+str(ts)+'.csv', index=False, encoding='utf-8')