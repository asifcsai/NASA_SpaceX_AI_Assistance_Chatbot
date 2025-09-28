import streamlit as st
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Configuration
os.environ['GROQ_API_KEY'] = 'replace_by your api'
os.environ['HUGGINGFACEHUB_API_KEY'] = 'replace by your api'

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = Path("faiss_index")

class SimpleRAGChatbot:
    def __init__(self):
        """Initialize the simplified RAG chatbot"""
        self.setup_components()
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    
    def setup_components(self):
        """Setup LLM, embeddings, and vector store"""
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1500
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Load vector store
        self.vector_store = self.load_vector_store()
        
        # Create retriever
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}
            )
    
    def load_vector_store(self):
        """Load the FAISS vector store"""
        try:
            if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
                vector_store = FAISS.load_local(
                    str(PERSIST_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return vector_store
            else:
                st.error("❌ FAISS index not found. Please run the indexing script first.")
                return None
        except Exception as e:
            st.error(f"❌ Error loading vector store: {str(e)}")
            return None
    
    def retrieve_context(self, question):
        """Retrieve relevant context for the question"""
        if not self.vector_store:
            return "No vector database available.", []
        
        try:
            # Get relevant documents
            retrieved_docs = self.retriever.invoke(question)
            
            # Format context
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get("source_file", "unknown")
                content = doc.page_content.strip()
                context_parts.append(f"Document {i} ({source}):\n{content}")
            
            formatted_context = "\n\n---\n\n".join(context_parts)
            return formatted_context, retrieved_docs
            
        except Exception as e:
            return f"Error retrieving context: {str(e)}", []
    
    def create_prompt(self, question, context, chat_history):
        """Create the prompt with system message, context, and question"""
        
        # System message
        system_msg = """You are an expert NASA SpaceX Research Assistant. You have access to a database of space research papers and documents.

INSTRUCTIONS:
1. Use the provided RESEARCH CONTEXT to answer questions accurately
2. If context is relevant, reference it specifically in your response
3. If context doesn't contain relevant info, say so and provide general knowledge
4. Be detailed and technical when appropriate
5. For greetings, respond warmly and explain your capabilities
6. Maintain conversation flow using chat history

RESPONSE STYLE:
- Professional yet friendly
- Use specific examples from the research when available
- Cite sources when referencing documents
- Provide step-by-step explanations for complex topics"""

        # Build messages
        messages = [SystemMessage(content=system_msg)]
        
        # Add chat history
        messages.extend(chat_history)
        
        # Create current message with context
        current_message = f"""RESEARCH CONTEXT:
{context}

USER QUESTION: {question}

Please provide a detailed answer using the research context above when relevant."""

        messages.append(HumanMessage(content=current_message))
        
        return messages
    
    def get_response(self, user_question):
        """Generate response using RAG"""
        try:
            # Step 1: Retrieve context
            context, retrieved_docs = self.retrieve_context(user_question)
            
            # Step 2: Create prompt with context
            messages = self.create_prompt(
                user_question, 
                context, 
                st.session_state.chat_history
            )
            
            # Step 3: Get LLM response
            response = self.llm.invoke(messages)
            response_content = response.content
            
            # Step 4: Update chat history
            st.session_state.chat_history.extend([
                HumanMessage(content=user_question),
                AIMessage(content=response_content)
            ])
            
            # Limit chat history to last 8 messages (4 exchanges)
            if len(st.session_state.chat_history) > 8:
                st.session_state.chat_history = st.session_state.chat_history[-8:]
            
            return response_content, retrieved_docs
            
        except Exception as e:
            return f"❌ Error: {str(e)}", []
    
    def clear_history(self):
        """Clear all chat history"""
        st.session_state.chat_history = []
        st.session_state.messages = []

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="NASA SpaceX Research Assistant", 
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 NASA SpaceX Research Assistant")
    st.markdown("*AI-powered assistant with access to space research database*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("🔄 Loading research assistant..."):
            st.session_state.chatbot = SimpleRAGChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            chatbot.clear_history()
            st.success("Chat cleared!")
            st.rerun()
        
        st.markdown("---")
        st.header("📊 Database Status")
        
        if chatbot.vector_store:
            try:
                num_docs = chatbot.vector_store.index.ntotal if hasattr(chatbot.vector_store.index, "ntotal") else "Unknown"
                st.success("✅ Database Connected")
                st.info(f"📄 Documents: {num_docs}")
            except:
                st.success("✅ Database Connected")
        else:
            st.error("❌ Database Not Found")
            st.info("Please run NASA_SpaceX.py first")
        
        st.markdown("---")
        st.header("💡 Try These Questions")
        
        sample_questions = [
            "What happens to mice in space?",
            "Tell me about microgravity effects",
            "How does space affect biological systems?", 
            "What are SpaceX's main achievements?",
            "Explain space radiation effects"
        ]
        
        for i, q in enumerate(sample_questions):
            if st.button(f"Q{i+1}: {q}", key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_question = q
    
    # Main chat area
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                        for i, doc in enumerate(message["sources"], 1):
                            source = doc.metadata.get("source_file", "unknown")
                            st.markdown(f"**{i}. {source}**")
                            st.text(doc.page_content[:300] + "...")
                            if i < len(message["sources"]):
                                st.markdown("---")
    
    # Handle sample question
    if 'sample_question' in st.session_state:
        user_input = st.session_state.sample_question
        del st.session_state.sample_question
    else:
        # Chat input
        user_input = st.chat_input("Ask about space research, missions, or experiments...")
    
    # Process user input
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching database and generating response..."):
                response, sources = chatbot.get_response(user_input)
            
            st.markdown(response)
            
            # Show sources in expandable section
            if sources:
                with st.expander(f"📚 Sources Used ({len(sources)} documents)"):
                    for i, doc in enumerate(sources, 1):
                        source = doc.metadata.get("source_file", "unknown")
                        chunk_id = doc.metadata.get("chunk_id", "")
                        
                        st.markdown(f"**{i}. {source}**")
                        if chunk_id:
                            st.caption(f"Chunk: {chunk_id}")
                        
                        # Show content in a text area
                        st.text_area(
                            f"Content {i}:",
                            doc.page_content,
                            height=100,
                            key=f"source_{i}",
                            disabled=True
                        )
                        
                        if i < len(sources):
                            st.markdown("---")
        
        # Add assistant response to messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    main()


