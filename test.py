import streamlit as st



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
with st.container(height=500):
    st.title("Echo Bot")
    messages = st.container(height=300)
# React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        messages.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = '''**Finetune Model**: The best matching page list is 216,215,208,214,204  
        **Base Model**: The best matching page list is 214,216,279,208,1  
        **Only Semantic**: The best matching page list is 214,279,1,10,174'''
        # Display assistant response in chat message container
        with messages.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})