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
import traceback

# ... (이전 코드는 그대로 유지)

def main():
    st.title("🤖Robot Conference Q&A System")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Report Mode"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    try:
        # ... (Pinecone 초기화 및 기타 설정은 그대로 유지)

        def get_chatbot_chain(prompt):
            answer = prompt | llm | StrOutputParser()
            return (
                RunnableParallel(
                    question=RunnablePassthrough(),
                    chat_history=RunnablePassthrough(),
                    docs=retriever
                )
                .assign(context=format)
                .assign(answer=answer)
                .pick("answer")
            )

        report_chain = get_report_chain(report_prompt)
        chatbot_chain = get_chatbot_chain(chatbot_prompt)

        # ... (모드 선택 및 디스플레이 로직은 그대로 유지)

        # User input
        if question := st.chat_input("Please ask a question about the conference:"):
            # Ensure the input is properly encoded
            question = question.encode('utf-8', errors='ignore').decode('utf-8')
            
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
                        
                        if st.session_state.mode == "Report Mode":
                            response = report_chain.invoke(question)
                        else:  # Chatbot Mode
                            # Format chat history
                            chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history])
                            # 수정된 부분: 딕셔너리 대신 문자열로 전달
                            response = chatbot_chain.invoke(question)
                        
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
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        st.error(f"Error details: {e.args}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        return
                    
                    finally:
                        # Clear status displays
                        status_placeholder.empty()
                        progress_bar.empty()
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display sources (only for Report Mode)
                    if st.session_state.mode == "Report Mode" and 'docs' in response:
                        with st.expander("Sources"):
                            for doc in response['docs']:
                                st.write(f"- {doc.metadata['source']}")
                    
                    # Add assistant's response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Update chat history for Chatbot Mode
                    if st.session_state.mode == "Chatbot Mode":
                        st.session_state.chat_history.append({"role": "human", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        # Limit chat history to last 10 messages
                        st.session_state.chat_history = st.session_state.chat_history[-10:]

        # Add a button to reset the conversation
        if st.button("Reset Conversation"):
            reset_conversation()

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Error details: {e.args}")
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
