import base64
from environment import model

def parse_image(image_data, ext="jpeg"):
    encoded = base64.b64encode(image_data).decode()
    response = model.invoke([{
        "role": "human",
        "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{encoded}"}
        }, {
            "type": "text",
            "text": "Describe this chart or diagram in detail, preserving all data and structure"
        }]
    }])
    return response.content

if __name__ == "__main__":
    from extract_images import extract_images_from_pdf
    images = extract_images_from_pdf("data/TS-SCERT-Class-10-Biology-Textbook-in-English-Medium.pdf")
    print(f"Found {len(images)} images")
    if images:
        img = images[0]
        print(f"Testing image from page {img['page']}, format={img['ext']}")
        caption = parse_image(img["data"], ext=img["ext"])
        print(f"Caption:\n{caption}")
