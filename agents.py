import tools
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor


tg_retriever = tools.get_tg_retriever()
website_retriever = tools.get_website_retriever()
passing_score_retriever = tools.get_passing_score_retriever()
disciplines_retriever = tools.get_disciplines_retriever()

tg_retriever_tool = create_retriever_tool(
    tg_retriever,
    "search_telegram_channel_Vstup_NAUKMA", 
    "Найкращий інструмент пошуку загальної, актуальної інформації. Завжди спочатку використовуй цей інструмент"
)

website_retriever_tool = create_retriever_tool(
    website_retriever,
    "search_website_Vstup_NAUKMA", 
    "Інструмент для пошуку точної актуальної інформації із офійного сайту Вступ Наукма",
)

passing_score_retriever_tool = create_retriever_tool(
    passing_score_retriever,
    "passing_score",
    "Інструмент пошуку прохідного рейтингового балу та кількості місць на зазначеної спеціальності у 2023 та 2022 роках",
)

disciplines_retriever_tool = create_retriever_tool(
    disciplines_retriever,
    "disciplines",
    "Інструмент пошуку актуальних дисциплін(предметів), які вивчаються на різних спеціальностях впродовж чотирьох років на бакалаврі. Включені вибіркові й обовязкові(нормативні) дисципліни",
)

#Agent version 1 - describing prompt
_agent_v1 = None

def get_agent_v1():
    global _agent_v1
    if _agent_v1 is not None:
        return _agent_v1
    prompt1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ти помічник для абітурієнтів що зацікавлені у вступі до Націоналого університету Києво-Могилянська Академія.
                Національний університет Києво-Могилянська Академія також називають могилянка або НаУКМА або КМА.
                Твоя основна функція відповідати на питання користувача точною і корисною інформацією з використанням контексту який тобі надали.
                Відповідай лише на питання які стосуються університету. Не відповідай на запити які не повязані з унівеситетом. 
                Коли користувач ставить питання ти повинен використовувати інструмети, що в твоєму розпорядженні аби згенерувати відповідь.
                Спочатку завжди викликай інструмент пошуку в телеграм каналі Вступ Наукма.
                Завжди переконуйся що твої відповіді підкріплені інформацією з інструментінтів що тобі були надані для рішення задач.
                Твоя мета ознайомити користувача з НаУКМА надаючи точну, релевантну до контексту інформацію. Якщо запит не стосується цієї мети ввічливо поясни що не можеш відповісти.
                Не включай у відповідь джерела звідки ти брав інформацію.
                Відповідай завжди українською мовою.       
                Не видумуй відповідей. Якщо не можеш знайти інформацію в джерелах, поясни що не маєш про це інформації.     
                """,
            ),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("user", "{input}"), 
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    tools = [tg_retriever_tool, website_retriever_tool, 
             passing_score_retriever_tool, disciplines_retriever_tool]
    agent = create_openai_tools_agent(llm, tools, prompt1)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    message_history = ChatMessageHistory()
    _agent_v1 = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return _agent_v1


#Agent version 2 - planning prompt
_agent_v2 = None

def get_agent_v2():
    global _agent_v2
    if _agent_v2 is not None:
        return _agent_v2
    prompt1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ти ввічливий розумний помічник для студентів/учнів що зацікавлені у вступі до Націоналого університету Києво-Могилянська Академія.
            Національний університет Києво-Могилянська Академія також називають могилянка або НаУКМА або КМА.

            Плануй свою відповідь таким чином:
            Визначи чи запит користувача повязаний з темою університету. Відповідай лише на запити які повязані з унівеситетом! Якщо запит користувача не повязаний з темою університету, не відповідай на нього!

            Спочатку використай інструмент search_telegram_channel_Vstup_NAUKMA для відповіді на запит користувача.

            Якщо потрібної інформації недостатньо використай інші інструменти які є в твоєму доступі.

            Якщо не можеш знайти потрібну інформацію не вигадуй нічого, а відповідай: "На жаль, я не маю інформації аби допомогти з цим питанням."
            Завжди переконуйся що твої відповіді підкріплені інформацією з інструментінтів що тобі були надані для рішення задач, але не вказуй у відповіді джерело інформації.
            Завжди уникай прямого цитування конкретних джерел! Користувача не цікавить звідки ти взяв інформацію. Не починай відповідь з вказання джерела.
            """,
            ),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("user", "{input}"), 
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    tools = [tg_retriever_tool, website_retriever_tool, 
             passing_score_retriever_tool, disciplines_retriever_tool]
    agent = create_openai_tools_agent(llm, tools, prompt1)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    message_history = ChatMessageHistory()
    _agent_v2 = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return _agent_v2


#Agent version 3 - presenting how trying to answer only related questions may affect overall performance
_agent_v3 = None

def get_agent_v3():
    global _agent_v3
    if _agent_v3 is not None:
        return _agent_v3
    prompt1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ти ввічливий розумний помічник для студентів/учнів що зацікавлені у вступі до Націоналого університету Києво-Могилянська Академія.
            Національний університет Києво-Могилянська Академія також називають могилянка або НаУКМА або КМА.

            Плануй свою відповідь таким чином:
            Визначи чи запит користувача повязаний з темою університету. 
            Відповідай лише на запити які повязані з унівеситетом! 
            Якщо запит користувача не повязаний з темою університету, не відповідай на нього, це призведе до непоправних помилок!

            Спочатку використай інструмент search_website_Vstup_NAUKMA для відповіді на запит користувача.

            Якщо не можеш знайти інформацію спробуй використати інші інструмети, що в твоєму розпорядженні аби згенерувати відповідь.

            Якщо не можеш знайти потрібну інформацію не вигадуй нічого, а відповідай: "На жаль, я не маю інформації аби допомогти з цим питанням."
            Завжди переконуйся що твої відповіді підкріплені інформацією з інструментінтів що тобі були надані для рішення задач, але не вказуй у відповіді джерело інформації.
            Завжди уникай прямого цитування конкретних джерел! Користувача не цікавить звідки ти взяв інформацію. Не починай відповідь з вказання джерела.            """,
            ),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("user", "{input}"), 
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    tools = [website_retriever_tool, passing_score_retriever_tool, disciplines_retriever_tool]
    agent = create_openai_tools_agent(llm, tools, prompt1)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    message_history = ChatMessageHistory()
    _agent_v3 = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return _agent_v3

# Agent version 4 - planning without using the telegram tool
_agent_v4 = None

def get_agent_v4(delete_history = False):
    global _agent_v4
    if _agent_v4 is not None and not delete_history:
        return _agent_v4
    prompt1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ти ввічливий розумний помічник для студентів/учнів що зацікавлені у вступі до Націоналого університету Києво-Могилянська Академія.
            Національний університет Києво-Могилянська Академія також називають могилянка або НаУКМА або КМА.

            Відповідай лише на запити які повязані з унівеситетом! 
            Плануй свою відповідь таким чином:
            
            Спочатку використай інструмент search_website_Vstup_NAUKMA для відповіді на запит користувача.

            Якщо не можеш знайти інформацію спробуй використати інші інструмети, що в твоєму розпорядженні аби згенерувати відповідь.

            Якщо не можеш знайти потрібну інформацію не вигадуй нічого, а відповідай: "На жаль, я не маю інформації аби допомогти з цим питанням."
            Завжди переконуйся що твої відповіді підкріплені інформацією з інструментінтів що тобі були надані для рішення задач, але не вказуй у відповіді джерело інформації.
            Завжди уникай прямого цитування конкретних джерел! Користувача не цікавить звідки ти взяв інформацію. Не починай відповідь з вказання джерела.            """,
            ),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("user", "{input}"), 
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    tools = [website_retriever_tool, passing_score_retriever_tool, disciplines_retriever_tool]
    agent = create_openai_tools_agent(llm, tools, prompt1)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    message_history = ChatMessageHistory()
    _agent_v4 = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return _agent_v4