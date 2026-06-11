import re
from langchain_core.documents import Document

def semantic_chunk(docs, max_chunk_size=500):
    #initiate chunk as list, later becomes list of dictionaries
    chunks = []
    # Step 1: split by headings first
    #in regex syntax, mark those with 1~3 #, headline fonts in MARKDOWN, as headline
    #OR up to 50 char, with 1+ '=' chars, which is underline beneath heading title of plain text docs
    heading_pattern = r'\n#{1,3} .+|\n[A-Z][^\n]{0,50}\n[-=]+'
    #split string into list, using regex heading_pattern specified above as separator
    sections = re.split(heading_pattern, docs)
    #searches through string and return list of all matches to heading_pattern (i.e., list of headings)
    headings = re.findall(heading_pattern, docs)
    #if no heading for one section, assume it is intro section and add section 
    if len(sections)>len(headings):
        headings.insert(0, "Introduction")
    #for index, section content split by heading, assume len(sections)=len(headings) now
    for i, section in enumerate(sections):
        heading = headings[i]
        # Step 2: if section is too long, split by paragraph
        #if long section compared to max_chunk_size
        if len(section) > max_chunk_size:
            #split section into paragraphs by two entries each
            paragraphs = section.split('\n\n')
            #for each split paragraph in section 
            for j, para, in enumerate(paragraphs):
                #if there is actual content
                if para.strip():
                    #add paragraph metadata as new dictionary item in chunks
                    chunks.append({
                        "text": para.strip(),
                        "section": heading.strip(),
                        "chunk_type": "paragraph", #saves chunk type as paragraph
                        "paragraph_index": j #saves paragraph number as paragraph index
                    })
        #if not long section compared to max_chunk_size
        else:
            #if section content not empty
            if section.strip():
                #add section metadata as new dictionary item in chunks
                chunks.append({
                    "text": section.strip(),
                    "section": heading.strip(),
                    "chunk_type": "section" #saves chunk type as section
                    #no paragraph_index
                })
    
    return chunks