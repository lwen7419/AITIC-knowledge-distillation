#import PyMuPDF, library for reading and manipulating pdf
import fitz

def extract_images_from_pdf(pdf_path):
    #open pdf and returns Document object that's iterable
    doc = fitz.open(pdf_path)
    #accumulator list holding one dict per image found
    images = []
    #enumerate over every page with index(page no.), page
    for page_num, page in enumerate(doc):
        #for each image in page
        #.get_images() is PyMuPDF method returning list 
        #of image unique id from that page
        for img in page.get_images():
            #set xref to unique ID for image object inside file
            xref = img[0]
            #takes xref and goes into pdf's object table
            #finds image object, decodes, and returns
            #dict with raw binary data representing pixels
            base_image = doc.extract_image(xref)
            #stores image bytes, page number, and image format
            #as dicts into images list
            images.append({
                "data": base_image["image"],
                "page": page_num,
                "ext": base_image["ext"]
            })
    return images