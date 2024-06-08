from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, status
from fastapi.responses import JSONResponse
from langchain_community.llms import LlamaCpp
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
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

from execute_excited_attention_model import get_important_features_and_correlated_features

model_path="./llms/Mistral-7B-Instruct-v0.3.Q8_0.gguf"
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

def perform_classification_fn(*args, **kwargs) -> dict[str, list[tuple]]:
    # Perform the classification
    result = get_important_features_and_correlated_features(app.state.df, app.state.dep_var)
    return result
custom_classification_tool = Tool.from_function(
    func=perform_classification_fn,
    #return_direct=True,
    name="Feature Importance Classifier",
    description="Use this tool to identify the top 5 most important features for classification given a dependent variable. \
        The tool will return a list of the top 5 most important features."
)

def drop_columns_fn(*args, **kwargs) -> list[str]:
    if args:
        columns_to_drop = args[0]
    else:
        columns_to_drop = []
    app.state.df.drop(columns=columns_to_drop, inplace=True)
    remaining_columns = app.state.df.columns.tolist()
    return remaining_columns
custom_drop_columns_tool = Tool.from_function(
    func=drop_columns_fn,
    name="Column Dropper",
    description="Use this tool to drop specified columns from a pandas dataframe. Provide a list of the column names to drop, \
        and the tool will return the names of the remaining columns in the dataframe."
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    app.state.llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1, 
        temperature=0.1, 
        max_tokens=4096, 
        n_ctx=4096,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )
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
    print("Chiamato execute")
    if not hasattr(request.app.state, 'csv_agent'):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded")

    result = request.app.state.csv_agent.invoke(query)
    return result['output']

@app.post("/start-model-and-get-first-review", tags=["Use Model"])
async def _start_model_get_first_review(file: UploadFile = File(...), dep_var: str = Form(...)):
    print("Chiamato start-model-and-get-first-review")
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
    result = app.state.csv_agent.invoke(f"Perform a classification on the dataframe with the dependent variable '{dep_var}'. \
                                        Use the Feature Importance Classifier tool to identify the top 5 most important features. \
                                        Provide a detailed response listing these features and explain that they were identified using the classification tool. \
                                        Ask me whatever you want me to do on the .csv file. For example, you can ask me to drop some columns from the .csv and restart the classification to determine the top 5 most important features.")
    return result['output']

@app.post("/debug-custom_classification_tool", tags=["Debug"])
async def _debug_custom_classification_tool(file: UploadFile = File(...), dep_var: str = Form(...)):
    df = pd.read_csv(file.file)
    app.state.df = df
    app.state.dep_var = dep_var
    result = perform_classification_fn(dep_var)
    return str(result)

@app.post("/debug-custom_drop_columns_tool", tags=["Debug"])
async def _debug_custom_drop_columns_tool(columns_to_drop: str = Form(...)):
    result = drop_columns_fn(columns_to_drop)
    return str(result)

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
    return templates.TemplateResponse(request, name="homepage.html")

@app.get("/wizard", tags=["Frontend"])
def progressbar(request: Request):

    return templates.TemplateResponse(request, name="wizard.html")

@app.get("/testbootstrap", tags=["Frontend"])
def testbootstrap(request: Request):
    return templates.TemplateResponse(request, name="testbootstrap.html")

@app.get("/csv", tags=["Frontend"])
def testbootstrap(request: Request):
    return templates.TemplateResponse(request, name="csv.html")

@app.get("/chat", tags=["Frontend"])
def chat(request: Request):
    return templates.TemplateResponse(request, name="chat.html")

@app.get("/chat_old", tags=["Frontend"])
def Step3(request: Request):
    return templates.TemplateResponse(request, name="chat_old.html")







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