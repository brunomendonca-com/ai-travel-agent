import os
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

import bs4


llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)

query = """
I'll travel to Sedona, Arizona, next week. What are some good places to visit \
and what events will be happening in that area? Include the cheapeast flight \
tickets price for non-stop flights from Calgary, Alberta, to Phoenix, AZ, \
and car rental. I'll be flying on the 19th and returning on the 24th.
"""


def researchAgent(query, llm):
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, prompt=prompt, verbose=True)
    webContext = agent_executor.invoke({"input": query})
    return webContext['output']


print(researchAgent(query, llm))


def loadData():
    loader = WebBaseLoader(
        web_path="https://visitsedona.com/things-to-do/100-things-to-do/",
        bs_kwargs=dict(parse_only=bs4.SoupStrainer("main"))
    )
    docs = loader.load()
    embedding = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorStore = Chroma.from_documents(
        documents=splits, embedding=embedding)
    retriever = vectorStore.as_retriever()
    return retriever


def getRelevantDocs(query):
    retriever = loadData()
    relevant_documents = retriever.invoke(query)
    print(relevant_documents)
    return relevant_documents


def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
        You're a travel agency manager. Your final answer should be a complete \
        and detailed travel itinerary for the client. The prices should be \
        accurate, converted to Canadian Dollars, and the events should be \
        relevant.

        Use the events and prices context, the user input, and also the \
        relevant documents to create the itinerary.

        Context: {webContext}
        Relevant Documents: {relevant_documents}
        User Input: {query}
        Assistant:
    """

    prompt = PromptTemplate(
        input_variables=[
            "webContext",
            "relevant_documents",
            "query"
        ],
        template=prompt_template
    )

    sequence = RunnableSequence(prompt, llm)

    response = sequence.invoke({
        "webContext": webContext,
        "relevant_documents": relevant_documents,
        "query": query
    })
    return response


def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response


print('\n', '-'*50, '\n')
print(getResponse(query, llm).content)
