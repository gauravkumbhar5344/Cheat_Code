from langchain.tools import tool
from langchain.agents import create_agent
import os
#from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

#load_dotenv()
llm = ChatGroq(
    api_key= "",
    model="openai/gpt-oss-120b",
    temperature=0
)
@tool
def blog_writer_agent(query:str)->str:
  """You are a professional Blog Writer.
     Given a topic, generate a full blog post.
  """
  prompt = f"""
    You are a professional blog writer.

    Topic: {query}

    Tasks:
    Before you write :
        - Know Your Audience: Understand who you're writing for, their problems, and what they want to learn.
        - Research & Niche: Choose a relevant topic, research it well, and consider using keyword research for SEO.
        - Define Your Goal: What action should the reader take (e.g., comment, sign up, learn)? 

    Writing the post :
        - Compelling Headline: Make it clear, enticing, and keyword-rich.
        - Engaging Intro: Grab attention immediately with a strong first sentence or opening paragraph.
        - Value-Driven Content: Provide depth, solve a problem, and offer unique insights or real-life experiences.
        - Structure & Formatting: Break text with subheadings, short paragraphs, bullet points, and images for easy scanning.
        - Clear Language: Use an easy-to-understand, conversational tone.
        - Internal & External Links: Add relevant links to other posts and authoritative sources.
        - Strong Conclusion & CTA: Summarize key points and tell the reader what to do next (comment, share, subscribe).

    Return only the blog content.
    """
  ai_msg=llm.invoke(prompt)
  return ai_msg

agent = create_agent(model=llm, tools=[blog_writer_agent])

user_input = input("Tell the topic : ")
result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

final_msg = result["messages"][-1]
print(getattr(final_msg, "content", final_msg))
