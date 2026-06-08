def semantic_chunk(text, max_chunk_size=500):
    chunks = []
    
    # Step 1: split by headings first
    import re
    heading_pattern = r'\n#{1,3} .+|\n[A-Z][^\n]{0,50}\n[-=]+'
    sections = re.split(heading_pattern, text)
    headings = re.findall(heading_pattern, text)
    
    for i, section in enumerate(sections):
        heading = headings[i] if i < len(headings) else "Introduction"
        
        # Step 2: if section is too long, split by paragraph
        if len(section) > max_chunk_size:
            paragraphs = section.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    chunks.append({
                        "text": para.strip(),
                        "section": heading.strip(),
                        "chunk_type": "paragraph"
                    })
        else:
            if section.strip():
                chunks.append({
                    "text": section.strip(),
                    "section": heading.strip(),
                    "chunk_type": "section"
                })
    
    return chunks