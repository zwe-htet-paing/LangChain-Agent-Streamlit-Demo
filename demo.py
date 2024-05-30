import os
import streamlit as st
from src.utils.agent import get_agent_executor

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


if __name__ == "__main__":
    agent_executor = get_agent_executor()

    # result = agent_executor.invoke({"input": "What is 23 plus 17?"})
    # print(result["output"])

    st.set_page_config(page_title="LangChain Agent Demo", page_icon="🦜🔗")
    st.title("💬 Search and Math Assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hi my name is Oliver and I am your personal consultant, how can I help you?",
            }
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is the capital of Myanmar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st.write("Thinking ...")
            st_callback = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=False
            )
            response = agent_executor.invoke(
                {"input": prompt}, {"callbacks": [st_callback]}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response["output"]}
            )
            st.write(response["output"])
