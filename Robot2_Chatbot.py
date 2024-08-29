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
                    'source': results['matches'][i]['metadata'].get('source', '').replace('C:\\Users\\minje\\data2\\', '') if 'source' in results['matches'][i]['metadata'] else 'Unknown'
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
    st.title("🤖Robot Conference Q&A System")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Report Mode"
    
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
    
    # Set up prompt template for report mode
    report_template = """
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
     Describe about 15,000+ words for covering industrial changes, issues, and response strategies related to the conference. Explicitly reflect and incorporate the [research principles] throughout your analysis and recommendations. 
    </description>

    <format>
     [Conference Overview]
        - Explain the overall context of the conference related to the question
        - Introduce the main points or topics
                   
     [Contents]
        - Analyze the key content discussed at the conference and reference.
        - For each key session or topic:
          - Gather the details as thoroughly as possible, then categorize them according to the following format: 
            - Topic : 
            - Fact : {{1. Provide a detailed description of approximately 5 sentences. 2. Include specific examples, data points, or case studies mentioned in the session. }}
            - Your opinion : {{Provide a detailed description of approximately 3 sentences.}}
            - Source : {{Show 2~3 data sources for each key topic}}
          
      [Conclusion]
        - Summarize new trends based on the conference content
        - Present derived insights
        - Suggest 3 follow-up questions that the LG Group representative might ask, and provide brief answers to each (3~4 sentences)
    </format>

    <style>Business writing with clear and concise sentences targeted at executives</style>

    <constraints>
        - USE THE PROVIDED CONTEXT TO ANSWER THE QUESTION
        - IF YOU DON'T KNOW THE ANSWER, ADMIT IT HONESTLY
        - ANSWER IN KOREAN AND PROVIDE RICH SENTENCES TO ENHANCE THE QUALITY OF THE ANSWER
        - ADHERE TO THE LENGTH CONSTRAINTS FOR EACH SECTION. [CONFERENCE OVERVIEW] ABOUT 4000 WORDS / [CONTENTS] ABOUT 7000 WORDS / [CONCLUSION] ABOUT 4000 WORDS
    </constraints>
    </task>
    </prompt>
    """
    report_prompt = ChatPromptTemplate.from_template(report_template)

    # Set up prompt template for chatbot mode
    chatbot_template = """
    Question: {question}
    Context: {context}
    Answer the question based on the given context in 4,000 words. 
            - Analyze the key content discussed at the conference and reference.
            - For each key session or topic:
                  - Gather the details as thoroughly as possible, then categorize them according to the following format: 
                  - Topic : 
                    - Fact : {{1. Provide a detailed description of approximately 5 sentences. 2. Include specific examples, data points, or case studies mentioned in the session. }}
                    - Your opinion : {{Provide a detailed description of approximately 3 sentences.}}
                    - Source : {{Show 2~3 data sources for each key topic}}
    If the answer is not in the context, say you don't know. Use a conversational tone and answer in Korean.
    """
    chatbot_prompt = ChatPromptTemplate.from_template(chatbot_template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source: {source}")
        return "\n\n" + "\n\n".join(formatted)

    format = itemgetter("docs") | RunnableLambda(format_docs)

    def get_report_chain(prompt):
        answer = prompt | llm | StrOutputParser()
        return (
            RunnableParallel(question=RunnablePassthrough(), docs=retriever)
            .assign(context=format)
            .assign(answer=answer)
            .pick(["answer", "docs"])
        )

    def get_chatbot_chain(prompt):
        answer = prompt | llm | StrOutputParser()
        return (
            RunnableParallel(question=RunnablePassthrough(), docs=retriever)
            .assign(context=format)
            .assign(answer=answer)
            .pick("answer")
        )

    report_chain = get_report_chain(report_prompt)
    chatbot_chain = get_chatbot_chain(chatbot_prompt)

    # Mode selection
    new_mode = st.radio("Select mode:", ("Report Mode", "Chatbot Mode"), key="mode_selection")
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = []  # Clear chat history when mode changes
        st.rerun()  # Changed from st.experimental_rerun()

    # Display current mode (for debugging)
    st.write(f"Current mode: {st.session_state.mode}")

    # Reset function
    def reset_conversation():
        st.session_state.messages = []
        st.rerun()  # Changed from st.experimental_rerun()

    # Check for reset keywords
    reset_keywords = ["처음으로", "초기화", "다시", "안녕"]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if question := st.chat_input("Please ask a question about the conference:"):
        # Check for reset keywords
        if any(keyword in question for keyword in reset_keywords):
            reset_conversation()
        else:
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
                    chain = report_chain if st.session_state.mode == "Report Mode" else chatbot_chain
                    response = chain.invoke(question)
                    time.sleep(1)  # Simulate search time
                    
                    # Step 3: Generating Answer
                    status_placeholder.text("Generating answer...")
                    progress_bar.progress(75)
                    answer = response['answer'] if st.session_state.mode == "Report Mode" else response
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
                
                # Display sources (only for Report Mode)
                if st.session_state.mode == "Report Mode":
                    with st.expander("Sources"):
                        for doc in response['docs']:
                            st.write(f"- {doc.metadata['source']}")
                
                # Add assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

    # Add a button to reset the conversation
    if st.button("Reset Conversation"):
        reset_conversation()

if __name__ == "__main__":
    main()
