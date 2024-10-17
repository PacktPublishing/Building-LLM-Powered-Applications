from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('mountain.txt') as f:
    mountain = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, #number of characters for each chunk
    chunk_overlap  = 20,#number of characters overlapping between a preceding and following chunk
    length_function = len #function used to measure the number of characters
)
texts = text_splitter.create_documents([mountain])
print(texts[0])
print(texts[1])
print(texts[2])