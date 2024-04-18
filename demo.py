import os
import streamlit as st
from src.utils.agent import get_agent_executor

from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)


st.set_page_config(page_title="🦜🔗 Demo Agent")
st.title('🦜🔗 Demo Agent')

agent_executor = get_agent_executor()

# if prompt := st.chat_input():
#     st.chat_message("user").write(prompt)
#     print(prompt)
#     with st.chat_message("assistant"):
#         st_callback = StreamlitCallbackHandler(st.container())
#         response = agent_executor.invoke(
#             {"input": prompt}, {"callbacks": [st_callback]}
#         )
#         st.write(response["output"])

st.title("💬 Search and Math Assistant")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi my name is Oliver and I am your personal consultant, how can I help you?"}]

#"st.session_state:", st.session_state.messages

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # with st.spinner('Preparing'):
    st_callback = StreamlitCallbackHandler(st.container())
    msg = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )

    #st.write(msg)
    #st.write(len(msg))
    #st.write(type(msg))

    st.session_state.messages.append({"role": "assistant", "content": msg["output"]})
    st.chat_message("assistant").write(msg["output"])