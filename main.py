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
    """ print("Chiamato execute")
    if not hasattr(request.app.state, 'csv_agent'):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded")

    result = request.app.state.csv_agent.invoke(query)
    return result['output'] """


async def _start_model_get_first_review(df, dep_var):
    msg = 'The top 5 most important features for classification are:\
    1. age (Importance Score: 0.5646992)\
    2. sex (Importance Score: 0.4274814)\
    3. restecg (Importance Score: 0.3393734)\
    4. fbs (Importance Score: 0.13357216)\
    5. trestbps (Importance Score: 0.11224261)\
    The top 5 most correlated features are:\
    1. age, cp (Correlation Score: 0.56581354)\
    2. sex, thal (Correlation Score: 0.55085963)\
    3. thalach, oldpeak (Correlation Score: 0.48816043)\
    4. age, oldpeak (Correlation Score: 0.48603684)\
    5. age, slope (Correlation Score: 0.48169476)]\
    Ask me whatever you want me to do on the .csv file. For example, you can ask me to drop some columns from the .csv and restart the classification to determine the top 5 most important features and correlations.'

    """ app.state.df = df
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
                                        Ask me whatever you want me to do on the .csv file. For example, you can ask me to drop some columns from the .csv and \
                                        restart the classification to determine the top 5 most important features.") """
    
    #await sio.emit("send_msg", result['output'])
    await sio.emit("send_resumee", msg)


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
@app.get("/test", tags=["Frontend"])
def test(request: Request):
    return templates.TemplateResponse(request, name="test.html")








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
    await _start_model_get_first_review(df, dep_var)

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
    resumee = 'The top 5 most important features for classification are:\
    1. age (Importance Score: 0.5646992)\
    2. sex (Importance Score: 0.4274814)\
    3. restecg (Importance Score: 0.3393734)\
    4. fbs (Importance Score: 0.13357216)\
    5. trestbps (Importance Score: 0.11224261)\
    The top 5 most correlated features are:\
    1. age, cp (Correlation Score: 0.56581354)\
    2. sex, thal (Correlation Score: 0.55085963)\
    3. thalach, oldpeak (Correlation Score: 0.48816043)\
    4. age, oldpeak (Correlation Score: 0.48603684)\
    5. age, slope (Correlation Score: 0.48169476)]\
    Ask me whatever you want me to do on the .csv file. For example,\
    you can ask me to drop some columns from the .csv and restart the classification\
    to determine the top 5 most important features and correlations.'

    await sio.emit("send_resumee", resumee)


@sio.on('question_model')
async def client_side_receive_msg(sid, msg):
    prompt1 = 'Now you have to drop columns from the .csv file, please drop the columns "sex" and "age"'
    prompt2 = 'Ok now restart the classification'
    answare = ''

    if msg == prompt1:
        answare = 'The .csv file now has the columns "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", and "target" remaining in it.'
        await sio.emit("model_answer", [answare, False])

    elif msg == prompt2:
        answare = 'The top 5 most important features for classification are:\
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
        await sio.emit("model_answer", [answare, True])

    else:
        answare = 'Mhh, let me think...'
        await sio.emit("model_answer", [answare, False])