from langchain_core.output_parsers import (
    JsonOutputParser,
    StrOutputParser,
)
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from datetime import datetime

from .schemas import AnswerQuestion, VerificationModel
from llms import llm


parser = JsonOutputParser(pydantic_object=AnswerQuestion)
verification_parser = JsonOutputParser(pydantic_object=VerificationModel)

boolean_parser = BooleanOutputParser(true_val="YES", false_val="NO")

follow_up_prompt = PromptTemplate(
    template="""You are an expert in conversation analysis. Your single task is to determine if the 'User Query' is a follow-up Question to the 'Previous Chat History'.

**Analysis Rules:**

1.  **Focus on the last exchange:** The most important clue is the relationship between the `User Query` and the very last message in the `Previous Chat History`.
    If the user is asking for any related data from chat history then it will be a follow up question.

2.  **Identify Follow-Up Intent:** A follow-up question directly refers to the subject of the last message. Look for intent to:
    - Request the same information in a different way.
    - A follow-up question asks for clarification, more detail, or elaborates on the immediately preceding topic.
    - A new question changes the subject or introduces a new incident.
    - Always check Whether the answer to the user query is found in previous chat history.

3.  **Identify New Questions:** A new question changes the subject entirely, introduces a new incident number, or asks about something not discussed in the last exchange.
    - If The Chat history is empty or no relevant data then classify it as new incident

**Your Response:**
If you found the user's query is a followup then respond with 'Yes' else 'No'.
Always Ensure not to mis predict. If the User's Query is related to any data or a small instance of the chat history classify it as followup.
You MUST respond with only the single word 'YES' or 'NO'. Do not provide any other text or explanation.

---
**Inputs for Analysis:**

**Previous Chat History:**
{chat_history}

**User Query:**
{query}
---
""",
    input_variables=["chat_history", "query"],
)

is_follow_up_chain = follow_up_prompt | llm | boolean_parser


# --- CORRECTED PROMPT FOR INITIAL ANSWER ---
# This chain's goal is to produce an AnswerQuestion object.
first_responder_prompt = ChatPromptTemplate.from_template(
    """You are an expert query analyst. Your task is to analyze the incoming 'Current Query' and determine the most efficient path to resolve it, using the 'Previous Chat History' for context.

        First, identify the nature of the 'Current Query':
        *   Is it a **direct user question**?
        *   Or is it a **system-generated reflection** asking for improvements on a previous answer?

        Next, based on this analysis, you MUST classify the required action into one of three categories: `HISTORIC`, `NEEDS_SEARCH`, or `CASUAL`.

        **Classification Rules:**

        3.  **`NEEDS_SEARCH`**:
            *   This is the default action.
            *   This includes any **system-generated reflection** query, as these always require a new search to improve the answer.
            *   It also includes any **direct user question** that introduces a new topic, a new incident number, or asks something that cannot be answered from the chat history.
            *   **Example Reflection Query:** "The answer is missing the resolution date for incident INC123."
            *   **Example New User Query:** "What is the status of ticket INC456?"

        1.  **`HISTORIC`**:
            *   Choose this if the 'Current Query' is a **direct user question** that asks for clarification, elaboration, or more detail about the topic discussed in the last turn of the 'Previous Chat History'.
            *   **Example User Query:** "Can you explain that in more detail?" after an incident summary was provided.

        2.  **`CASUAL`**:
            *   Choose this if the 'Current Query' is a **direct user question** that is a simple greeting, thank-you, or conversational filler.
            *   **Example User Query:** "Thanks for the help!"


        ---
        **Inputs for Analysis:**
        
        **Is it a follow up query"**
        {is_follow_up}

        **Current Query:**
        {query}

        **Previous Chat History:**
        {chat_history}

        
        You MUST format your entire response as a JSON object that strictly follows the provided schema.
        {format_instructions}""",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# This chain now correctly produces an AnswerQuestion object
first_responder = first_responder_prompt | llm | parser


# CASUAL WORKFLOW CHAIN___________________________________________
casual_prompt = ChatPromptTemplate.from_template(
    """You are a friendly and helpful assistant. Provide a short, conversational response to the user's query. Ensure not to add any reflection steps or comments. return only the final casual answer:
        Chat History\n
        {chat_history}
        Query\n
        {query}"""
)

casual_response_chain = casual_prompt | llm | StrOutputParser()


# History Aware chain________________________________
history_aware_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert assistant. Your task is to answer the user's question based on the provided conversation history. Analyze the entire history to understand the context fully. Provide a detailed and comprehensive answer. If the answer isn't in the history, say so clearly.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

history_aware_chain = history_aware_prompt | llm | StrOutputParser()

# --- NEW PROMPT FOR Finalizing THE ANSWER ---
# This chain's goal is to produce a ReviseAnswer object.
revisor_prompt = ChatPromptTemplate.from_template(
    """You are an expert Incident Resolver. Your task is to synthesize information from the conversation history and new research to provide a final, comprehensive answer to the user's query.

            Follow these instructions carefully:
            1.  **Review the Conversation History:** Understand the user's previous questions and the answers given so far. This provides the full context.
            2.  **Focus on the User's Latest Query:** Identify the specific question the user is asking now.
            3.  **Use the Research Context:** The provided 'Research Context' contains new information retrieved to answer the user's query. You MUST base your final answer on this information.
            4.  **Synthesize and Answer:** Formulate a new, improved, and detailed answer. Directly address the user's latest query, integrating the information from the research context.
            5.  **Cite Your Sources:** Where you use information from the research context, cite the corresponding reference number directly in your answer (e.g., "The system was updated on Tuesday [1].").

            **User's Query** 
            {query}

            **Chat History**
            {chat_history}
            
            ---
            **Research Context:**
            {references}
            ---
            
            Based on our conversation and the new research, please provide a final, detailed answer.
            """
)

second_responder = revisor_prompt | llm | StrOutputParser()

verification_prompt = PromptTemplate(
    template="""
    You are a meticulous quality assurance expert. Your task is to verify if the provided 'Generated Answer' addresses the 'User Query'.
    
    - If the answer contains the bare minimum information for user's query, set 'is_sufficient' to True.
    - If the answer is insufficient, set 'is_sufficient' to False and provide a clear 'reflection' on what needs to be improved. The reflection should be a direct instruction for improvement.
    
    User Query: {query}\n\nGenerated Answer: {answer}
    
    **Output Instructions:**
    Provide a valid JSON object following this format:
    {format_instructions}
    """,
    input_variables=["query", "answer"],
    partial_variables={
        "format_instructions": verification_parser.get_format_instructions()
    },
)

verifier_chain = verification_prompt | llm | verification_parser
