#converts binary data (images, audio) into ASCII charaters
#travel through text-based system without corruption
import base64
from environment import model

def parse_image(image_data, ext="jpeg"):
    #convert raw image bytes into base64 bytes
    #now bytes object but with ASCII char in it
    #.decode() converts from bytes into Python string
    encoded = base64.b64encode(image_data).decode()
    #langchain calls llm imported from environment
    #with images as base64 encoded string representation
    response = model.invoke([{
        #set role to human to specify that this is user input
        "role": "human",
        #list with two items to be sent to model
        "content": 
        #set to image_url which signals to langchain
        # that block contains image and the image_url is dict 
        # with url key holding the actual image
        [{
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{encoded}"}
        },
        #second item is text prompt with type set to text signaling
        #  prompt is plain text and text holds actual prompt string telling 
        # model what to do with the image
        {
            "type": "text",
            "text": "Describe this chart or diagram in detail, preserving all data and structure"
        }]
    }])
    return response.content

if __name__ == "__main__":
    from extract_images import extract_images_from_pdf
    images = extract_images_from_pdf("data/TS-SCERT-Class-10-Biology-Textbook-in-English-Medium.pdf")
    if images:
        img = images[0]
        print(f"Testing image from page {img['page']}, format={img['ext']}")
        caption = parse_image(img["data"], ext=img["ext"])
        print(f"Caption:\n{caption}")