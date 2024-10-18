import os
import streamlit as st
from utils import save_feedback
import time
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated,List
from langgraph.graph import END
from langchain_cerebras import ChatCerebras
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]["API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "Short Video Theme Researcher"

# enhanced streamlit chatbot interface
st.sidebar.header("✏️ Short Video Theme Researcher")
st.sidebar.markdown(
    "This app helps you find trending short video themes based on your interests. "
    "To use this app, you'll need to provide a Cerebras API key, which you can obtain for free [here](https://cloud.cerebras.ai/platform/org_nxh29kc28dt5rvrcphxv54et/apikeys) "
    "and a Tavily Search key, which you can obtain [here](https://app.tavily.com/home)."
)
st.sidebar.write("### Instructions")
st.sidebar.write(":pencil: Enter a video idea you would like to explore.")
st.sidebar.write(":point_right: Click 'Generate Suggestions' to receive a detailed discussion on the topic.")
st.sidebar.write(":heart_decoration: Let me know your thoughts and feedback about the app.")

# Initialize session state for feedback storage if not already done
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

# Feedback form
st.sidebar.subheader("Feedback Form")
feedback = st.sidebar.text_area("Your thoughts and feedback", value=st.session_state.feedback, placeholder="Share your feedback here...")

if st.sidebar.button("Submit Feedback"):
    if feedback:
        try:
            save_feedback(feedback)
            st.session_state.feedback = ""  # Clear feedback after submission
            st.sidebar.success("Thank you for your feedback! 😊")
        except Exception as e:
            st.sidebar.error(f"Error saving feedback: {str(e)}")
    else:
        st.sidebar.error("Please enter your feedback before submitting.")

st.sidebar.image("assets/logo01.jpg", use_column_width=True)

final_result = []

# Ask user for their Gemini API key via `st.text_input`.
Cerebras_api_key = st.text_input("Cerebras API Key", type="password", placeholder="Your Cerebras API key here...")
Tavily_api_key = st.text_input("Tavily API Key", type="password", placeholder="Your Tavily API key here...")

if not Cerebras_api_key or not Tavily_api_key:
    st.info("Please add your Cerebras and Tavily to continue.", icon="🗝️")
else:
    os.environ["TAVILY_API_KEY"] = Tavily_api_key
    class State(TypedDict):
        query: Annotated[list, add_messages]
        subqueries: List[str]
        optimized_query: List[str]
        research: Annotated[list, add_messages]
        content: str
        content_ready: bool
        iteration_count: int
        # Counter for iterations

    # Initialize ChatCerebras instance for language model
    llm = ChatCerebras(api_key=Cerebras_api_key, model="llama3.1-70b")
    # llm = ChatGroq(api_key=Cerebras_api_key, model="llama-3.1-70b-versatile")

    class ResearchAgent:
        def __init__(self):
            self.tavily_tool = TavilySearchResults(
                max_results=3,
                include_answer=True,
                include_raw_content=True,
                include_images=True,
            )

        def format_search(self, query: str) -> str:
            """
            Takes the user query and returns a list of optimized subqueries.
            If the query is complex, breaks it down into subqueries.
            """
            prompt = (
                "You are an expert at breaking down user's query into subqueries if needed and optimizing these search queries for Google. "
                "Your task is to take a given query, break it down into step by step thought process and return an optimized version of it, making it more likely to yield relevant results. "
                "Do not include any explanations or extra text, only the optimized query.\n\n"
                "The breakdown of subqueries should be limited to a maximum of three subqueries"
                "Example:\n"
                "Original: How to create a successful AI-powered content strategy\n"
                "Optimized: What are the best AI tools for content creation\n\n"
                "Optimized: How to use [specific AI tool] to develop a content strategy\n\n"
                "Optimized: How to measure the success of an AI-powered content strategy\n\n"
                "Example:\n"
                "Original: How do I generate leads using content marketing\n"
                "Optimized: How to use blog posts for lead generation\n\n"
                "Optimized: Generating leads through video content marketing\n\n"
                "Optimized: Using downloadable content for lead generation\n\n"
                "Example:\n"
                "Original: compare cats and dogs eating habits\n"
                "Optimized: eating habits of cats\n"
                "Optimized: eating habits of dogs\n\n"
                "The breakdown of subqueries should be limited to a maximum of three subqueries. Now optimize the following query:\n"
                f"Original: {query}\n"
                "Optimized:"
            )

            # Call LLM to process the prompt and return optimized subqueries
            response = llm.invoke(prompt)

            # Split the response content into a list of subqueries
            optimized_subqueries = response.content.splitlines()
            optimized_subqueries = [subquery for subquery in optimized_subqueries if subquery.strip()]
        
            return optimized_subqueries
            
      
        def search(self, state: State):
            """
            Handles multiple searches if the query is broken into subqueries.
            Aggregates the results from each subquery into the final content.
            """
            start_time = time.perf_counter()

            # Get the last user query and format it
            user_query = state.get('query', "")[-1].content
            optimized_queries = self.format_search(user_query)
            state["optimized_query"] = optimized_queries  # Store optimized subqueries in state

            final_results = []
            for i, subquery in enumerate(optimized_queries):
                results = self.tavily_tool.invoke({'query': subquery})
                formatted_results = self.format_results(results)
                final_results.append(f"Subquery {i+1}: {subquery}\n{formatted_results}\n")

            end_time = time.perf_counter()

            # Aggregated formatted results
            aggregated_results = "\n".join(final_results)

            # Create a valid AIMessage with 'role' and 'content'
            research_message = AIMessage(role="assistant", content=aggregated_results)

            # Update the research in state with the valid AIMessage
            state["research"].append(research_message)

            # Return the result in the required format for the next step
            return {"research": [research_message]}

        def format_results(self, results):
            """
            Formats the search results from Tavily tool.
            """
            formatted = "Search Results:\n\n"
            for item in results:
                formatted += f"Title: {item.get('title', 'N/A')}\n"
                formatted += f"URL: {item.get('url', 'N/A')}\n"
                formatted += f"Content: {item.get('content', 'N/A')}\n\n"
            return formatted

    class EditorAgent:
        def evaluate_research(self, state: State):
            query = '\n'.join(message.content for message in state.get("query"))
            research = '\n'.join(message.content for message in state.get("research"))

            iteration_count = state.get("iteration_count", 1)
            
            if iteration_count is None:
                iteration_count = 1
            
            if iteration_count >= 3:
                return {"content_ready": True}
            
            prompt = (
                "You are an expert editor. Your task is to evaluate the research based on the query. "
                "If the information is sufficient to create a comprehensive and accurate blog post, respond with 'sufficient'. "
                "If the information is not sufficient, respond with 'insufficient' and provide a new, creative query suggestion to improve the results. "
                "If the research results appear repetitive or not diverse enough, think about a very different kind of question that could yield more varied and relevant information. "
                "Consider the depth, relevance, and completeness of the information when making your decision.\n\n"
                "Example 1:\n"
                "Used queries: What are the benefits of a Mediterranean diet?\n"
                "Research: The Mediterranean diet includes fruits, vegetables, whole grains, and healthy fats.\n"
                "Evaluation: Insufficient\n"
                "New query: Detailed health benefits of a Mediterranean diet\n\n"
                "Example 2:\n"
                "Used queries: How does solar power work?\n"
                "Research: Solar power works by converting sunlight into electricity using photovoltaic cells.\n"
                "Evaluation: Sufficient\n\n"
                "Example 3:\n"
                "Used queries: Effects of climate change on polar bears?\n"
                "Research: Climate change is reducing sea ice, affecting polar bear habitats.\n"
                "Evaluation: Insufficient\n"
                "New query: How are polar bears adapting to the loss of sea ice due to climate change?\n\n"
                "Now evaluate the following:\n"
                f"Used queries: {query}\n"
                f"Research: {research}\n\n"
                "Evaluation (sufficient/insufficient):\n"
                "New query (if insufficient):"
            )
            
            start_time = time.perf_counter()
            response = llm.invoke(prompt)
            end_time = time.perf_counter()

            evaluation = response.content.strip()

            final_result.append({"subheader": f"Editor Evaluation Iteration", "content": evaluation, "time": end_time - start_time})

            if "new query:" in evaluation.lower():
                new_query = evaluation.split("New query:", 1)[-1].strip()
                return {"query": [new_query], "iteration_count": iteration_count + 1, "evaluation": evaluation}
            else:
                return {"content_ready": True, "evaluation": evaluation}
            
    class WriterAgent:
        def write_blogpost(self, state: State):
            query = state.get("query")[0].content
            research = '\n'.join(message.content for message in state.get("research"))

            prompt = (
                "You are an expert blog post writer. Your task is to take a given query and context, and write a comprehensive, engaging, and informative short blog post about it. "
                "Make sure to include an introduction, main body with detailed information, and a conclusion. Use a friendly and accessible tone, and ensure the content is well-structured and easy to read.\n\n"
                f"Query: {query}\n\n"
                f"Context:\n{research}\n\n"
                "Write a detailed and engaging blog post based on the above query and context."
            )

            response  = llm.invoke(prompt)

            return {"content": response.content}

    # Initialize the StateGraph
    graph = StateGraph(State)

    graph.add_node("search_agent", ResearchAgent().search)
    graph.add_node("writer_agent", WriterAgent().write_blogpost)
    graph.add_node("editor_agent", EditorAgent().evaluate_research)

    graph.set_entry_point("search_agent")

    graph.add_edge("search_agent", "editor_agent")

    graph.add_conditional_edges(
        "editor_agent",
        lambda state: "accept" if state.get("content_ready") else "revise",
        {
            "accept": "writer_agent",
            "revise": "search_agent"
        }
    )

    graph.add_edge("writer_agent", END)

    graph = graph.compile()

    user_input = st.text_input("Enter a video idea you would like to explore", placeholder="challenges of raising a Montessori child")

    if st.button("Generate output"):
        if not user_input:
            st.error("Please fill out all fields to generate a output.", icon="🚫")
        else:
            with st.spinner("Generating detailed discussion..."):
                start_time = time.perf_counter()
                blogpost = graph.invoke({"query": user_input})
                end_time = time.perf_counter()

            # Display final blog post
            st.subheader("A Detailed Discussion")
            st.write(f"Time taken: {end_time - start_time:.2f} seconds")
            st.write(blogpost["content"])
