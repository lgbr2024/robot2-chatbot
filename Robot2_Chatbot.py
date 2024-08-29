import streamlit as st
import os
from dotenv import load_dotenv
from operator import itemgetter
from typing import List, Tuple, Dict, Any
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]

class ModifiedPineconeVectorStore(PineconeVectorStore):
    def __init__(self, index, embedding, text_key: str = "text", namespace: str = ""):
        super().__init__(index, embedding, text_key, namespace)
        self.index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 8, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Tuple[Document, float]]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True,
            include_values=True,
            filter=filter,
            namespace=namespace,
        )
        return [
            (
                Document(
                    page_content=result["metadata"].get(self._text_key, ""),
                    metadata={k: v for k, v in result["metadata"].items() if k != self._text_key}
                ),
                result["score"],
            )
            for result in results["matches"]
        ]

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 8, fetch_k: int = 30,
        lambda_mult: float = 0.7, filter: Dict[str, Any] = None, namespace: str = None
    ) -> List[Document]:
        namespace = namespace or self._namespace
        results = self.index.query(
            vector=embedding,
            top_k=fetch_k,
            include_metadata=True,
            include_values=True,
            filter=filter,
            namespace=namespace,
        )
        if not results['matches']:
            return []
        
        embeddings = [match['values'] for match in results['matches']]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=min(k, len(results['matches'])),
            lambda_mult=lambda_mult
        )
        
        return [
            Document(
                page_content=results['matches'][i]['metadata'].get(self._text_key, ""),
                metadata={
                    'source': results['matches'][i]['metadata'].get('source', '').replace('C:\\Users\\minje\\data3\\', '') if 'source' in results['matches'][i]['metadata'] else 'Unknown'
                }
            )
            for i in mmr_selected
        ]

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    k: int = 4,
    lambda_mult: float = 0.5
) -> List[int]:
    similarity_scores = cosine_similarity([query_embedding], embedding_list)[0]
    selected_indices = []
    candidate_indices = list(range(len(embedding_list)))
    for _ in range(k):
        if not candidate_indices:
            break
        
        mmr_scores = [
            lambda_mult * similarity_scores[i] - (1 - lambda_mult) * max(
                [cosine_similarity([embedding_list[i]], [embedding_list[s]])[0][0] for s in selected_indices] or [0]
            )
            for i in candidate_indices
        ]
        max_index = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(max_index)
        candidate_indices.remove(max_index)
    return selected_indices

def main():
    st.title("Robot2 Conference Q&A System")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "robot"
    index = pc.Index(index_name)
    
    # Select GPT model
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4o"
    
    st.session_state.gpt_model = st.selectbox("Select GPT model:", ("gpt-4o", "gpt-4o-mini"), index=("gpt-4o", "gpt-4o-mini").index(st.session_state.gpt_model))
    llm = ChatOpenAI(model=st.session_state.gpt_model)
    
    # Set up Pinecone vector store
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )
    
    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    # Set up prompt template and chain
    template = """
    <prompt>
        Question: {question} 
        Context: {context} 
        Answer:
        <context>
            <role>Strategic consultant for LG Group, tasked with uncovering new trends and insights based on various conference trends.</role>
            <audience>
                -LG Group individual business executives
                -LG Group representative
            </audience>
            <knowledge_base>Conference file saved in vector database</knowledge_base>
            <goal>Find and provide organized content related to the conference that matches the questioner's inquiry, along with sources, to help derive project insights.</goal>
        </context>
        <task>
            <description>
                Analyze and provide insights on industrial changes, issues, and response strategies related to the conference, following a phased approach. Explicitly reflect and incorporate the [research principles] throughout your analysis and recommendations.
            </description>
            <format>
                <phase name="Preparation">
                    - Confirm the conference-related question from the user
                    - If not provided, ask "Which conference are you interested in?"
                </phase>
           <phase name="1. Conference Overview and Key Points">
                - Summarize conference information based on vector DB data
                - Present a table with conference name and summary
                - Analyze core content related to user's specific questions
                - For each key topic:
                    - Key Points:
                    - Business Cases:
                    - Promising Application Areas/Methods:
                    - Sources: (Provide 2-3 data sources)
                - Ask if user has additional questions
            </phase>
            
            <phase name="2. Implications and Analysis">
                - Provide insights based on user questions and conference content
                - Separate (Fact) and (Opinion)
                - Include relevant conference content and sources
                - Summarize new trends based on the conference content
                - Present derived insights
                - Suggest 3 follow-up questions that the LG Group representative might ask, and provide brief answers to each (3-4 sentences)
            </phase>
        </format>
        <style>Business writing with clear and concise sentences targeted at executives</style>
        <constraints>
            - USE THE PROVIDED CONTEXT TO ANSWER THE QUESTION
            - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
            - ANSWER IN KOREAN AND PROVIDE RICH SENTENCES TO ENHANCE THE QUALITY OF THE ANSWER
            - INFORM THE USER WHEN ENTERING A NEW PHASE AND CONFIRM BEFORE PROCEEDING
            - ALLOW THE USER TO RETURN TO PREVIOUS PHASES IF DESIRED
            - ADHERE TO THE LENGTH CONSTRAINTS: PHASE 1 ABOUT 7000 WORDS / PHASE 2 ABOUT 8000 WORDS
        </constraints>
    </task>
    <phases>
        <general_instructions>
            - Address all phases.
            - Each phase has specific goals to achieve.
            - Inform the user when they have reached the goal state and confirm if they want to proceed to the next phase.
            - Notify the user when entering a new phase (e.g., ## Phase 1. Conference Overview and Key Points).
            - Allow the user to return to previous phases if desired.
        </general_instructions>
        <preparation_phase>
            <goal>The user provides a "conference-related question" necessary for analysis.</goal>
            <evaluation_criteria>
                Confirm if the user has successfully provided a "conference-related question". If not provided properly, repeatedly request with the message "Which conference are you interested in?"
            </evaluation_criteria>
        </preparation_phase>
        <phase1>
            <name>Conference Overview and Key Points</name>
            <goal>Summarize conference information and analyze key points based on the user's questions.</goal>
            <additional_instructions>
                - Organize conference name and [summary] content in a table
                - Analyze and synthesize key points, business cases, and promising application areas/methods
                - Always provide sources at the end of each key topic
            </additional_instructions>
            <evaluation_criteria>
                Confirm if the user has additional questions or is ready to proceed to the implications phase.
            </evaluation_criteria>
        </phase1>
        <phase2>
            <name>Implications and Analysis</name>
            <goal>Provide implications and in-depth analysis based on the conference content and user's questions.</goal>
            <additional_instructions>
                - Prioritize writing implications based on user questions
                - Separate (Fact) and (Opinion) in the writing
                - Include relevant conference content and sources
                - Summarize new trends and present derived insights
                - Suggest follow-up questions with brief answers
            </additional_instructions>
        </phase2>
    </phases>
    </prompt>
     """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)

    format = itemgetter("docs") | RunnableLambda(format_docs)
    answer = prompt | llm | StrOutputParser()
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if question := st.chat_input("Please ask a question about the conference:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            # Create placeholders for status updates
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Query Processing
                status_placeholder.text("Processing query...")
                progress_bar.progress(25)
                time.sleep(1)  # Simulate processing time
                
                # Step 2: Searching Database
                status_placeholder.text("Searching database...")
                progress_bar.progress(50)
                response = chain.invoke(question)
                time.sleep(1)  # Simulate search time
                
                # Step 3: Generating Answer
                status_placeholder.text("Generating answer...")
                progress_bar.progress(75)
                answer = response['answer']
                time.sleep(1)  # Simulate generation time
                
                # Step 4: Finalizing Response
                status_placeholder.text("Finalizing response...")
                progress_bar.progress(100)
                time.sleep(0.5)  # Short pause to show completion
                
            finally:
                # Clear status displays
                status_placeholder.empty()
                progress_bar.empty()
            
            # Display the answer
            st.markdown(answer)
            
            # Display sources
            with st.expander("Sources"):
                for doc in response['docs']:
                    st.write(f"- {doc.metadata['source']}")
            
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
