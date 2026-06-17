#converts binary data (images, audio) into ASCII charaters
#travel through text-based system without corruption
import base64
#load Langchain ollama connector, now send messages to 
#locally running model
from langchain_ollama import ChatOllama
from environment import model

def parse_image(image_data):
    #convert raw image bytes into base64 bytes
    #now bytes object but with ASCII char in it
    #.decode() converts from bytes into Python string
    encoded = base64.b64encode(image_data).decode()
    #langchain calls llm imported from environment
    #with images as base64 encoded string representation
    response = model.invoke([{
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{encoded}"
    }, {
        "type": "text",
        "text": "Describe this chart or diagram in detail, preserving all data and structure"
    }])
    #extract actual text the model generated
    return response.content