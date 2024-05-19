import PyPDF2
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Loading environment variables from .env file
load_dotenv() 

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.environ['GROQ_API_KEY']

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
            groq_api_key=groq_api_key, model_name="llama3-70b-8192",
                         temperature=0.2)


@cl.on_chat_start
async def on_chat_start():
    files = None #Initialize variable to store uploaded files

    await cl.Avatar(
        name="bot",
        url="https://i.ibb.co/ngTMWqF/chatbot-avatar.jpg",
    ).send()

    await cl.Avatar(
        name="user",
        url="https://i.ibb.co/26cjmx7/360-F-87281963-29bnk-FXa6-RQn-JYWe-Rfr-Spieag-Nxw1-Rru.jpg",
    ).send()

    # Wait for the user to upload files
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=1000,# Optionally limit the file size,
            max_files=10,
            timeout=180, # Set a timeout for user response,
        ).send()

    # Process each uploaded file
    texts = []
    metadatas = []
    for file in files:
        print(file) # Print the file object for debugging

        # Read the PDF file
        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create a metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)


    HF_TOKEN = os.environ['HF_TOKEN']
    # Create a Chroma vector store
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="mixedbread-ai/mxbai-embed-large-v1")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    # Sending an image with the number of files
    elements = [
    ]
    # Inform the user that processing has ended.You can now chat.
    msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!",elements=elements,author="bot")
    await msg.send()

    #store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb],author="user")
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name,display='side')
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    #return results
    await cl.Message(content=answer, elements=text_elements,author="bot").send()