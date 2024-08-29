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
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Ensure current_phase is always initialized
    if "current_phase" not in st.session_state:
        st.session_state.current_phase = "PHASE 1"
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "robot"
    index = pc.Index(index_name)
    
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4o"
    
    st.session_state.gpt_model = st.selectbox("Select GPT model:", ("gpt-4o", "gpt-4o-mini"), index=("gpt-4o", "gpt-4o-mini").index(st.session_state.gpt_model))
    llm = ChatOpenAI(model=st.session_state.gpt_model)
    
    vectorstore = ModifiedPineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        text_key="source"
    )
    
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
    )
    
    template = """
    <prompt>
        <initial_greeting>
            안녕하세요. LG경영연구원의 Chatbot입니다. 어떤 컨퍼런스나 주제에 대해 분석해 드릴까요? 특정 주제에 대한 구체적인 질문이 있으시다면 말씀해 주세요. 그렇지 않다면 전반적인 컨퍼런스 분석을 진행하겠습니다.
        </initial_greeting>
        
        Question: {question}
        Context: {context}
        Current Phase: {current_phase}
        
        <response_logic>
            <if_phase_1>
                ##PHASE 1. 컨퍼런스 찾기
                - 목표: 사용자의 질문과 일치하는 컨퍼런스를 찾습니다.
                - 평가 기준: 컨퍼런스의 주요 주제, 날짜, 주요 기업 사례에 대한 간략한 개요로 시작합니다. 
                - 반드시 사용자에게 "이 컨퍼런스가 찾으시는 것이 맞나요? 맞다면 '네'라고 해주시고, 아니라면 어떤 정보가 필요한지 말씀해 주세요."라고 물어봐야 합니다.
                - 만약 사용자가 구체적인 질문을 했다면, 해당 질문에 대한 간단한 답변을 제공한 후 위의 질문을 해주세요.
            </if_phase_1>
            <if_phase_2>
                ##PHASE 2. 컨퍼런스 세부정보 제공
                - 목표: 컨퍼런스에 대한 약 8,000자의 설명을 제공합니다.
                - 다음 구조를 따르되, 질문의 성격에 따라 유연하게 조정하세요:
                    1. 질문 요약 및 배경
                    2. 관련 컨퍼런스 데이터 분석 **분석을 뒷받침할 구체적인 기업 사례, 숫자, 전문가 의견을 포함하고 대답 마지막에 출처를 포함하여 신뢰성을 높이세요.**
                    3. 주요 인사이트 및 트렌드
            </if_phase_2>
        </response_logic>

        <detailed_prompt>
            [이전과 동일한 내용 유지]
        </detailed_prompt>

        Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', '알 수 없는 출처')
            formatted.append(f"출처: {source}")
        return "\n\n" + "\n\n".join(formatted)

    format = itemgetter("docs") | RunnableLambda(format_docs)
    answer = prompt | llm | StrOutputParser()
    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            docs=retriever,
            current_phase=lambda x: st.session_state.get("current_phase", "PHASE 1")
        )
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer"])
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        initial_greeting = "안녕하세요. LG경영연구원의 Chatbot입니다. 어떤 컨퍼런스나 주제에 대해 분석해 드릴까요? 특정 주제에 대한 구체적인 질문이 있으시다면 말씀해 주세요. 그렇지 않다면 전반적인 컨퍼런스 분석을 진행하겠습니다."
        with st.chat_message("assistant"):
            st.markdown(initial_greeting)
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    
    if question := st.chat_input("컨퍼런스에 대해 질문해 주세요:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_placeholder.text("쿼리 처리 중...")
                progress_bar.progress(25)
                time.sleep(1)
                
                status_placeholder.text("데이터베이스 검색 중...")
                progress_bar.progress(50)
                response = chain.invoke(question)
                time.sleep(1)
                
                status_placeholder.text("답변 생성 중...")
                progress_bar.progress(75)
                answer = response['answer']
                time.sleep(1)
                
                status_placeholder.text("응답 마무리 중...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"죄송합니다. 응답을 생성하는 도중 오류가 발생했습니다: {str(e)}")
                return
            finally:
                status_placeholder.empty()
                progress_bar.empty()
            
            st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            if st.session_state.current_phase == "PHASE 1" and "네" in question.lower():
                st.session_state.current_phase = "PHASE 2"
                st.info("PHASE 2로 넘어갑니다. 컨퍼런스에 대한 세부 정보를 제공하겠습니다.")
            
            # PHASE 2에서 추가 질문 버튼 추가
            if st.session_state.current_phase == "PHASE 2":
                if st.button("추가 질문하기"):
                    st.session_state.current_phase = "PHASE 1"
                    st.info("새로운 질문을 위해 PHASE 1로 돌아갑니다.")

if __name__ == "__main__":
    main()
