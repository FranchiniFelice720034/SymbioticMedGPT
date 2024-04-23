from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from langchain_community.llms import LlamaCpp
from langchain_experimental.agents import create_csv_agent
model_path="./llms/mistral-7b-v0.1.Q5_0.gguf"

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5173/"
]

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
