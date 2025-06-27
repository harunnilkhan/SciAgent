from typing import Dict, Any, List, Tuple, Annotated, TypedDict, Literal, Optional
import json
import operator
import logging
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from sci_agent.config import TEMPERATURE, MAX_ITERATIONS
from sci_agent.chain import get_llm, create_qa_chain, create_summary_chain
from sci_agent.retriever import retrieve_relevant_documents, format_retrieved_documents

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Tool and state definitions
class ActionType(str, Enum):
    SEARCH = "search"
    ANSWER = "answer"
    SUMMARIZE = "summarize"
    FINAL_ANSWER = "final_answer"


class AgentState(TypedDict):
    """Type definition for the agent state"""
    messages: List[Dict[str, Any]]
    query: str
    documents: List[Dict[str, Any]]
    collection_name: str
    action: Optional[ActionType]
    answer: str


class Tool(BaseModel):
    """Model for tool definition"""
    name: str
    description: str

    def run(self, input_text: str) -> str:
        """
        Run the tool.

        Args:
            input_text: Input text

        Returns:
            str: Output text
        """
        raise NotImplementedError("Must be implemented by subclasses")


class SearchTool(Tool):
    """Document search tool"""
    name: str = "search"
    description: str = "Used to search for information in scientific papers"
    collection_name: str = Field(default="")

    def run(self, input_text: str) -> str:
        """Search documents"""
        try:
            docs = retrieve_relevant_documents(input_text, self.collection_name)
            return format_retrieved_documents(docs)
        except Exception as e:
            logger.error(f"Search tool error: {e}")
            return f"An error occurred during search: {str(e)}"


class AnswerTool(Tool):
    """Question answering tool"""
    name: str = "answer"
    description: str = "Used to answer the user's question"
    collection_name: str = Field(default="")

    def run(self, input_text: str) -> str:
        """Answer the question"""
        try:
            qa_chain = create_qa_chain(self.collection_name)
            return qa_chain.invoke(input_text)
        except Exception as e:
            logger.error(f"Answer tool error: {e}")
            return f"An error occurred while answering the question: {str(e)}"


class SummarizeTool(Tool):
    """Summarization tool"""
    name: str = "summarize"
    description: str = "Used to summarize the paper"
    collection_name: str = Field(default="")

    def run(self, input_text: str) -> str:
        """Summarize the paper"""
        try:
            summary_chain = create_summary_chain(self.collection_name)
            return summary_chain.invoke("")
        except Exception as e:
            logger.error(f"Summarization tool error: {e}")
            return f"An error occurred during summarization: {str(e)}"


def create_agent_prompt():
    """
    Creates the system prompt for the agent.

    Returns:
        str: System prompt
    """
    return """
    You are SciAgent, a scientific paper analysis assistant.

    To handle the user's questions or requests about scientific papers, follow these steps:

    1. When the user asks a question, first understand it correctly and determine what information you need.
    2. To gather relevant information, you can search the documents.
    3. Use the appropriate tools to create summaries or answer detailed questions.
    4. When composing your response, indicate the sources of the information you used and cite them properly.

    Available actions:
    - search: to search for relevant information in the documents
    - answer: to answer the user's specific question
    - summarize: to create a summary of the paper
    - final_answer: to provide your final response

    At each step, choose an action and respond in the following format:

    ```json
    {
      "action": "your chosen action",
      "action_input": "input for the action (text)"
    }
    ```

    Choose your actions wisely and provide comprehensive and accurate information to the user.
    """


def create_tools_config(collection_name: str) -> Tuple[List[Tool], Dict[str, Any]]:
    """
    Creates the agent's tools and configuration.

    Args:
        collection_name: Name of the vector database collection

    Returns:
        Tuple: List of tools and tool configuration
    """
    # Create the tools
    search_tool = SearchTool(collection_name=collection_name)
    answer_tool = AnswerTool(collection_name=collection_name)
    summarize_tool = SummarizeTool(collection_name=collection_name)

    tools = [search_tool, answer_tool, summarize_tool]

    # Tool executor
    def tool_executor(tool_name: str, tool_input: str) -> str:
        for tool in tools:
            if tool.name == tool_name:
                return tool.run(tool_input)
        return f"Tool not found: {tool_name}"

    return tools, {"execute": tool_executor}


def agent_node(state: AgentState, tool_executor, llm):
    """
    The main logic node of the agent.

    Args:
        state: Current agent state
        tool_executor: Tool executor
        llm: Language model

    Returns:
        Dict: Updated agent state
    """
    # Prepare messages
    messages = state["messages"]

    # Call the LLM
    try:
        response = llm.invoke(messages)

        # Parse the response
        content = response

        # Find the JSON format
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content and "}" in content:
            # Find the JSON block
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        else:
            # If JSON format not found, treat as final_answer
            return {"action": ActionType.FINAL_ANSWER, "answer": content}

        # Parse the JSON
        try:
            parsed = json.loads(json_str)
            action = parsed.get("action", "final_answer")
            action_input = parsed.get("action_input", "")

            # Convert to ActionType
            try:
                action = ActionType(action)
            except ValueError:
                action = ActionType.FINAL_ANSWER

            # Return the action and its input
            return {"action": action, "query": action_input}
        except json.JSONDecodeError:
            # If JSON cannot be parsed, treat as final_answer
            return {"action": ActionType.FINAL_ANSWER, "answer": content}

    except Exception as e:
        logger.error(f"Agent node error: {e}")
        # In case of error, return as final answer
        return {"action": ActionType.FINAL_ANSWER, "answer": f"An error occurred while generating the response: {str(e)}"}


def tool_node(state: AgentState, tool_executor):
    """
    Tool execution node.

    Args:
        state: Current agent state
        tool_executor: Tool executor

    Returns:
        Dict: Updated agent state
    """
    action = state["action"]

    if action == ActionType.FINAL_ANSWER:
        return {"answer": state.get("answer", "Sorry, I could not find an answer.")}

    # Get the tool input
    query = state.get("query", "")

    # Execute the tool
    try:
        result = tool_executor["execute"](action.value, query)

        # Append the result to messages
        tool_message = AIMessage(content=result)

        messages = state.get("messages", [])
        messages.append(tool_message)

        return {"messages": messages}

    except Exception as e:
        logger.error(f"Tool node error: {e}")
        messages = state.get("messages", [])
        messages.append(AIMessage(content=f"An error occurred while executing the tool: {str(e)}"))
        return {"messages": messages}


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Determines whether the agent should continue.

    Args:
        state: Current agent state

    Returns:
        str: "continue" or "end"
    """
    action = state.get("action")

    if action == ActionType.FINAL_ANSWER:
        return "end"

    # Check message count (to prevent infinite loops)
    messages = state.get("messages", [])
    if len(messages) > MAX_ITERATIONS * 2:  # Each iteration includes 2 messages (human and AI)
        return "end"

    return "continue"


def create_agent(collection_name: str):
    """
    Builds the SciAgent workflow graph.

    Args:
        collection_name: Vector database collection name

    Returns:
        StateGraph: Compiled agent graph
    """
    # Initialize the LLM model
    llm = get_llm()

    # Create the system prompt
    system_prompt = create_agent_prompt()

    # Create tools and configuration
    tools, tool_executor = create_tools_config(collection_name)

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes - simplified LLM handling for Ollama
    workflow.add_node(
        "agent",
        lambda state: agent_node(state, tool_executor, llm)
    )
    workflow.add_node("tools", lambda state: tool_node(state, tool_executor))

    # Add edges
    workflow.add_edge("agent", "tools")
    workflow.add_conditional_edges(
        "tools",
        should_continue,
        {
            "continue": "agent",
            "end": END
        }
    )

    # Set the entry point
    workflow.set_entry_point("agent")

    # Compile the graph
    return workflow.compile()


def run_agent(query: str, collection_name: str):
    """
    Runs the agent and returns its response.

    Args:
        query: User query
        collection_name: Vector database collection name

    Returns:
        str: Agent's response
    """
    try:
        agent = create_agent(collection_name)

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "documents": [],
            "collection_name": collection_name,
            "action": None,
            "answer": ""
        }

        result = agent.invoke(initial_state)

        return result.get("answer", "Sorry, a response could not be generated.")

    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return f"An error occurred while running the agent: {str(e)}"
