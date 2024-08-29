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
        <role>You are a strategic consultant for LG Group, tasked with uncovering new trends and insights based on various conference trends.</role>
        <audience>LG Group individual business executives and representatives</audience>
        <knowledge_base>Conference information stored in a vector database</knowledge_base>
        <goal>Provide comprehensive analysis and insights related to the specified conference, aligning with the inquirer's interests and LG Group's strategic needs.</goal>
    </context>
    
    <task>
        <description>
            Conduct a thorough analysis of the specified conference, focusing on industrial changes, emerging issues, and potential response strategies. Your analysis should be presented in a clear, professional manner suitable for high-level executives. Ensure that your response is well-structured, insightful, and directly relevant to LG Group's interests.
        </description>
    
        <format>
            Your response should be organized into two main sections:
    
            1. Conference Analysis and Key Insights
            Begin with a brief overview of the conference, including its name, date, and primary focus. Then, delve into the key topics discussed, highlighting their relevance to LG Group. For each significant point:
            - Provide a concise explanation of the topic or trend
            - Discuss any notable business cases or examples presented
            - Analyze potential applications or implications for LG Group
            - Include relevant data points or expert opinions to support your analysis
            
            2. Strategic Implications and Recommendations
            Based on the conference content and your analysis:
            - Identify the most significant trends or developments for LG Group
            - Discuss potential impacts on LG's current business models or strategies
            - Propose strategic recommendations or areas for further exploration
            - Suggest follow-up actions or research topics for LG Group to consider
    
            Throughout your response, ensure that you:
            - Maintain a professional, analytical tone
            - Use clear, concise language appropriate for executive-level readers
            - Support your points with specific examples or data from the conference
            - Clearly distinguish between factual information from the conference and your own analysis or opinions
        </format>
    
        <style>
            Write in a professional business style, using clear and concise sentences. Your tone should be analytical and authoritative, appropriate for high-level executive readers. Use paragraphs to structure your thoughts, and incorporate bullet points or numbered lists where appropriate to enhance readability.
        </style>
    
        <constraints>
            - Base your analysis solely on the information provided in the context about the conference
            - If certain information is not available, acknowledge this limitation in your response
            - Provide your answer in Korean, ensuring high-quality, nuanced language use
            - Aim for a comprehensive response of approximately 15,000 words, with about 60% dedicated to the Conference Analysis and 40% to Strategic Implications
            - Include relevant sources or references where applicable
        </constraints>
    </task>
    
    <phases>
        <phase1>
            <name>Conference Analysis and Key Insights</name>
            <goal>Provide a comprehensive overview and analysis of the conference content.</goal>
            <instructions>
                Begin by confirming the specific conference of interest with the user. Then, analyze the conference data to present key topics, trends, and notable discussions. Structure this information in a logical, easy-to-follow format, ensuring that each point is relevant to LG Group's interests.
            </instructions>
        </phase1>
    
        <phase2>
            <name>Strategic Implications and Recommendations</name>
            <goal>Synthesize the conference insights into actionable strategies for LG Group.</goal>
            <instructions>
                Based on the analysis from Phase 1, identify the most significant implications for LG Group. Provide strategic recommendations and suggest areas for further exploration or action. Ensure that your suggestions are practical, forward-thinking, and aligned with LG Group's business objectives.
            </instructions>
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
