import os
from openai import AsyncOpenAI
import chainlit as cl
from dotenv import load_dotenv
from swarms import Agent
from swarms.utils.function_caller_model import OpenAIFunctionCaller
from pydantic import BaseModel, Field
from swarms.structs.conversation import Conversation

# Import prompts
from prompts import (
    MASTER_AGENT_SYS_PROMPT,
    SUPERVISOR_AGENT_SYS_PROMPT,
    COUNSELOR_AGENT_SYS_PROMPT,
    BUDDY_AGENT_SYS_PROMPT
)

# RAG imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import tiktoken
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CallLog(BaseModel):
    agent_name: str = Field(description="The name of the agent to call: either Counselor-Agent or Buddy-Agent")
    task: str = Field(description="The task for the selected agent to handle")

# Initialize RAG Components
async def initialize_rag():
    try:
        logging.info("Starting initialize_rag")
        file_id = "1Co1QBoPlWUfSShlS8Evw8e6t1PHtAPGT"
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logging.info(f"Attempting to load document from: {direct_url}")
        docs = PyMuPDFLoader(direct_url).load()
        logging.info(f"Successfully loaded document, got {len(docs)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            length_function=lambda x: len(tiktoken.encoding_for_model("gpt-4").encode(x))
        )
        split_chunks = text_splitter.split_documents(docs)
        logging.info(f"Split into {len(split_chunks)} chunks")
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        qdrant_vectorstore = Qdrant.from_documents(
            split_chunks,
            embedding_model,
            location=":memory:",
            collection_name="knowledge_base",
        )
        
        return qdrant_vectorstore.as_retriever()
        
    except Exception as e:
        logging.error(f"Error in initialize_rag: {str(e)}")
        raise e

class EnhancedAgent(Agent):
    def __init__(self, agent_name, system_prompt, retriever):
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            max_loops=1,
            model_name="gpt-4"
        )
        self.retriever = retriever
    
    async def process_with_context(self, task: str) -> str:
        try:
            # Get context from RAG using ainvoke instead of aget_relevant_documents
            context_docs = await self.retriever.ainvoke(task)
            context = "\n".join([doc.page_content for doc in context_docs])
            
            enhanced_task = f"""
            Context from knowledge base:
            {context}
            
            User query:
            {task}
            """
            
            # Critical change: Don't await self.run
            return self.run(task=enhanced_task)
            
        except Exception as e:
            logging.error(f"Error in process_with_context: {str(e)}")
            return str(e)  # Return error message instead of raising

class SwarmWithRAG:
    def __init__(self, retriever):
        self.master_agent = OpenAIFunctionCaller(
            base_model=CallLog,
            system_prompt=MASTER_AGENT_SYS_PROMPT
        )
        
        self.counselor_agent = EnhancedAgent(
            agent_name="Counselor-Agent",
            system_prompt=COUNSELOR_AGENT_SYS_PROMPT,
            retriever=retriever
        )
        
        self.buddy_agent = EnhancedAgent(
            agent_name="Buddy-Agent",
            system_prompt=BUDDY_AGENT_SYS_PROMPT,
            retriever=retriever
        )
        
        self.agents = {
            "Counselor-Agent": self.counselor_agent,
            "Buddy-Agent": self.buddy_agent
        }

    async def process(self, message: str) -> str:
        try:
            # Get agent selection
            function_call = self.master_agent.run(message)
            agent = self.agents.get(function_call.agent_name)
            
            if not agent:
                return f"No agent found for {function_call.agent_name}"
            
            # Process with context - note we're awaiting here
            response = await agent.process_with_context(function_call.task)
            return response
            
        except Exception as e:
            logging.error(f"Error in process: {str(e)}")
            return f"Error processing your request: {str(e)}"
        
@cl.on_chat_start
async def start_chat():
    try:
        retriever = await initialize_rag()
        swarm = SwarmWithRAG(retriever)
        cl.user_session.set("swarm", swarm)
        
        await cl.Message(
            content="Hello! I'm SARASWATI, your AI guidance system. I have both a counselor and a buddy agent ready to help you. What would you like to discuss?"
        ).send()
        
    except Exception as e:
        error_msg = f"Error initializing chat: {str(e)}"
        logging.error(error_msg)
        await cl.Message(
            content=f"System initialization error: {error_msg}"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        swarm = cl.user_session.get("swarm")
        response = await swarm.process(message.content)
        await cl.Message(content=response).send()
        
    except Exception as e:
        logging.error(f"Error in message processing: {str(e)}")
        await cl.Message(
            content="I apologize, but I encountered an error. Please try again."
        ).send()

if __name__ == "__main__":
    cl.run()