import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import agents

load_dotenv()

if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

st.set_page_config(page_title="Vstup NAUKMA chatbot", page_icon="⭐")
st.title('Vstup NAUKMA chatbot')

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.chat_history = []
        chatbot_agent = agents.get_agent_v4(True)

for message in st.session_state.chat_history:
  if isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.markdown(message.content)
  else:
    with st.chat_message("AI"):
      st.markdown(message.content)


user_query = st.chat_input("Enter your question")

chatbot_agent = agents.get_agent_v4()

if user_query is not None and user_query != "":
  st.session_state.chat_history.append(HumanMessage(user_query))

  with st.chat_message("Human"):
    st.markdown(user_query)

  with st.chat_message("AI"):
    ai_response = chatbot_agent.invoke({"input": user_query}, 
                                       config={"configurable": {"session_id": "<foo>1"}})
    st.markdown(ai_response["output"])
    
  st.session_state.chat_history.append(AIMessage(ai_response["output"]))