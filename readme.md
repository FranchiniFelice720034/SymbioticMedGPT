# MedGPT CSV Execution API
This project implements a FastAPI to execute queries on a CSV dataset using the LlamaCpp Mistral-7B model.

## Prerequisites
To use this API, you need Python 3.8 or higher. All project dependencies are listed in requirements.txt.

## Installation
Follow these steps to set up the project locally.

## Create Python Environment
python -m venv env
env\Scripts\Activate.ps1


### Install Dependencies
Install the necessary Python dependencies with:

```
pip install -r requirements.txt
```

### Download the Model
1. Download the LlamaCpp Mistral-7B model from Hugging Face:
    - Visit the model page: [Mistral-7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/tree/main)
    - Download the model file.
2. Create a folder named llms in the main directory of your project.
3. Move the downloaded model file into the llms folder.

### Start the API
To start the API server, use the following command:
```
uvicorn main:app --reload
```

The --reload flag is useful for development as it allows the server to automatically restart with every code change.

### Usage
Go to http://localhost:8000/docs and try it!