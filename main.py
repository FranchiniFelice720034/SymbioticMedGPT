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
from env import MODEL_PATH 
import random
import socketio
import re
import ast
import asyncio
from fastapi.concurrency import run_in_threadpool

from execute_excited_attention_model import get_important_features_and_correlated_features

model_path=MODEL_PATH
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
    name="Feature Importance and Correlation Classifier",
    description="Use this tool to identify the top 5 most important features for classification given a dependent variable, \
        along with their importance scores, and the top 5 feature correlations. \
        The tool will return a dictionary with two keys: 'top_5_features' and 'top_5_correlations'. \
        'top_5_features' will be a list of tuples, where each tuple contains a feature name and its importance score. \
        'top_5_correlations' will be a list of tuples, where each tuple contains two feature names and their correlation score."
)

def drop_columns_fn(*args, **kwargs) -> list[str]:
    if args:
        if isinstance(args[0], pd.DataFrame) or (isinstance(args[0], str) and "[" not in args[0]):
            columns_to_drop = args[1] if len(args) > 1 else []
        else:
            columns_to_drop = args[0]

        if isinstance(columns_to_drop, str):
            columns_to_drop = columns_to_drop.replace(' ', '').replace('[', '').replace(']', '').replace("'", '').replace('"', '').replace("\\", '')
            print(columns_to_drop)
            columns_to_drop = columns_to_drop.split(',')
        if not isinstance(columns_to_drop, list):
            raise TypeError("Expected a list or a string representation of a list.")
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

async def _execute_model(query: str):
    def sync_execute_model(query):
        print("Chiamato execute")
        if not hasattr(app.state, 'csv_agent'):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded")

        if "classification" in query.lower():
            idx = query.lower().index("classification")
            query = (
                query[:idx + len("classification")] +
                ". \
                Use the Feature Importance and Correlation Classifier tool to identify the top 5 most important features and the top 5 most correlated features. \
                Use as action \"Feature Importance and Correlation Classifier\", and as input \"df, target='target'\"" +
                query[idx + len("classification"):]
            )

            query += " Provide a detailed response listing these features and correlations, explaining that they were identified using the classification tool. " \
                     "Provide a numbered list for both the top 5 most important features and their importance scores, and the top 5 most correlated features along with their correlation scores. " \
                     "End the response saying: Ask me whatever you want me to do on the .csv file. For example, you can ask me to drop some columns from the .csv and restart the classification to determine the top 5 most important features and correlations."

        print("-----------------------")
        print("Query: ", query)
        print("-----------------------")
        result = app.state.csv_agent.invoke(query)
        return result['output']

    result = await run_in_threadpool(sync_execute_model, query)
    return result


async def _start_model_get_first_review():
    def sync_start_model_get_first_review():
        app.state.csv_agent = create_pandas_dataframe_agent(
            app.state.llm, app.state.df, include_df_in_prompt=True, 
            prefix=PREFIX, verbose=True,
            extra_tools=[custom_classification_tool, custom_drop_columns_tool],
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "input_variables":['df_head', 'input', 'agent_scratchpad', 'chat_history_buffer'],
                "memory": chat_history_buffer
            }
        )
        query = f"Perform a classification on the dataframe with the dependent variable '{app.state.dep_var}'. \
                Use the Feature Importance and Correlation Classifier tool to identify the top 5 most important features and the top 5 most correlated features. \
                Your action must be \"Feature Importance and Correlation Classifier\", and the input must be \"df, target='target'\" \
                Provide a detailed response listing these features and correlations, explaining that they were identified using the classification tool. \
                Provide a numbered list for both the top 5 most important features and their importance scores, and the top 5 most correlated features along with their correlation scores. \
                end the response saying someting like: Ask me whatever you want me to do on the .csv file. For example, you can ask me to drop some columns from the .csv and restart the classification to determine the top 5 most important features and correlations."
        print("-----------------------")
        print("Query: ", query)
        print("-----------------------")
        result = app.state.csv_agent.invoke(query)
        return result['output']

    result = await run_in_threadpool(sync_start_model_get_first_review)
    return result


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

@app.get("/home")
def home(request: Request):
    left = 'Fernando'
    right = 'Nicotera'
    return templates.TemplateResponse(request, name="home.html", context={'id1': left, 'id2':right})


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
@app.get("/test", tags=["Frontend"])
def test(request: Request):
    return templates.TemplateResponse(request, name="test.html")








class ObjectListItem(BaseModel):
    item: str

@app.post("/senddata", tags=["Frontend"])
async def get_body(request: Request):

    print("\n\n\nChiamato senddata\n\n\n")
    data = await request.json()
    df = pd.json_normalize(data['table'])
    dep_var = data['dep_var']
    header = df.loc[0, :].values.tolist()
    df = df.iloc[1: ]
    
    df.columns = header

    if(not os.path.isdir(csv_uploaded_path)):
        os.mkdir(csv_uploaded_path)
    
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y-%H-%M-%S')
    print(df)

    d = '&nbsp'
    if d in dep_var:
        dep_var = dep_var.replace(d, ' ')    
    
    print(dep_var)
    df.to_csv('csv/uploaded/csv_'+str(ts)+'.csv', index=False, encoding='utf-8')

    app.state.df = df
    app.state.dep_var = dep_var

#Socket
sio=socketio.AsyncServer(cors_allowed_origins='*',async_mode='asgi')
#wrap with ASGI application
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

@sio.on("connect")
async def connect(sid, env):
    print("New Client Connected to This id :"+" "+str(sid))
    await sio.emit("send_msg", "Hello from Server")
@sio.on("disconnect")
async def disconnect(sid):
    print("Client Disconnected: "+" "+str(sid))

@sio.on('request_resumee')
async def client_side_receive_msg(sid, msg):
    output = await _start_model_get_first_review()
    await sio.emit("send_resumee", output)

@sio.on('question_model')
async def client_side_receive_msg(sid, msg):
    '''prompt1 = 'Now you have to drop columns from the .csv file, please drop the columns "sex" and "age"'
    prompt2 = 'Ok now restart the classification'
    answer = ''

    if msg == prompt1:
        answer = 'The .csv file now has the columns "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", and "target" remaining in it.'
        await sio.emit("model_answer", [answer, False])

    elif msg == prompt2:
        answer = 'The top 5 most important features for classification are:\
                            1. slope (Importance Score: 0.34682822)\
                            2. oldpeak (Importance Score: 0.3028454)\
                            3. fbs (Importance Score: 0.09012242)\
                            4. thal (Importance Score: 0.080464534)\
                            5. chol (Importance Score: -0.078735836)\
                            The top 5 most correlated features are:\
                            1. oldpeak, slope (Correlation Score: 0.5361485)\
                            2. trestbps, thalach (Correlation Score: 0.5220156)\
                            3. slope, thalach (Correlation Score: 0.49823856)\
                            4. slope, exang (Correlation Score: 0.48489332)\
                            5. thalach, oldpeak (Correlation Score: 0.47241843)'
        await sio.emit("model_answer", [answer, True])

    else:
        answer = 'Mhh, let me think...'
        await sio.emit("model_answer", [answer, False])'''
    
    output = await _execute_model(msg)
    await sio.emit("model_answer", [output, False])

    #Domanda1: Now you have to drop columns from the .csv file, please drop the columns "sex" and "age"
    #Domanda2: Ok now restart the classification
    