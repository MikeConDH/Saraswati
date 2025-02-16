# Saraswati
![download](https://github.com/user-attachments/assets/6359fec9-b784-45d5-a025-442cb8bd6b62)

## Purpose
Saraswati is an AI-driven system designed to support families navigating the stressful process of a high school student's college or university application. More than just an educational tool, Saraswati acts as a trusted mentor, coach, and companion, helping students explore their options and build confidence for the challenges of higher education.

## Actors
Student: High school preparing for college admission

Parent: The student’s parent

eCounciler: ChatBot acting as an effective counselor.

Buddy: ChatBot acting as an online counselor and therapist.


## Tech
Swarms: facilitating the orchestration for a multple agent system

OpenAI: embedding and foundation models: gpt-4o

Qdrant: vector database

Fetch.ai: deploy to internet in agentverse

Hugging Face: Docker container in Spaces for demo

## Spec

All the tech details are availabe in the spec:
https://docs.google.com/document/d/1phv68LTMdK9XpiVKLp1MX7rTov8w3oPWjwGrsu4AlZU/edit?usp=sharing

## QuickStart - Git R Done: 

1. Clone the repo
   
2. Use UV to set up the enviorment:

      uv venv .venv

      source .venv/bin/activate
   
3. Set up .env with
   
      WORKSPACE_DIR="agent_workspace"

      OPENAI_API_KEY="YOUR API KEY"
   
      ANTHROPIC_API_KEY="YOUR API KEY"
   
5) Run the app local using Chainlit
   
      (.venv) uv run chainlit run rag_saraswati.py -w
   
6) Test in web brower.  

Lets Go!




