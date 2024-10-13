import streamlit as st
import pandas as pd
import os
import re
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
from custom_tools import initialize_tools, get_data
from callbacks import StreamHandler, stream_data
from config import llm

# Set the page layout to wide
st.set_page_config(layout="wide")

# url = os.getenv("CSV_URL")
# file_id=url.split('/')[-2]
# dwn_url='https://drive.google.com/uc?id=' + file_id
# data_input = pd.read_csv(dwn_url)


# Initialize tools
tools = initialize_tools()

# Define the prefix and suffix for the prompt
prefix = '''You are the best Data Analyst that can help me Financial Analyst in summarizing findings, drawing insights, and making recommendations from a financial product segment data.
You should not make things up and only answer all questions related to the data and product segment.

STAR AM Company Background:
PT Surya Timur Alam Raya Asset Management (STAR AM) is a company engaged in asset management in Indonesia. The company has a primary focus on investment fund management and investment advisory, both for individuals and business entities. STAR AM is dedicated to always providing quality investment solutions, which allows STAR AM to assert its position as one of the investment managers with a high level of development in the mutual fund industry in Indonesia.

definition of channel columns 'client_type' in data_input:
- APERD: APERD is an abbreviation for Agen Pedagang Efek Reksa Dana (Mutual Fund Securities Selling Agent). The agent offers mutual fund investment services in a practical manner through online or digital applications that are easy to use by both individual or collective (group) investors.
- INSTO: INSTO refers to a direct sales channel for mutual fund or investment fund subscriptions. This channel is specifically designed to cater to institutional clients, which can include companies, organizations, and high-net-worth individuals. Unlike retail channels, which serve everyday individual investors, the INSTO channel typically deals with larger investment volumes and more complex financial structures

definition of product columns 'fund_type' in data_input:
- Balanced: Balanced Fund is a type of mutual fund designed to provide balanced growth in its investments. The main goal of this mutual fund is to create a balanced portfolio between financial instruments, such as bonds or debt securities, and riskier instruments like stocks. This way, investors can enjoy a more evenly distributed potential for investment growth.
- Fixed Income: FIXED INCOME FUND aims to provide optimal investment returns over the long term by investing in debt securities issued by the Government of the Republic of Indonesia and/or Indonesian corporations that have been sold through Public Offerings and/or traded on the Indonesia Stock Exchange, as well as domestic money market instruments.
- Equity: a type of mutual fund or pooled investment vehicle that primarily invests in stocks (also known as equities) of publicly traded companies. The goal of an equity fund is to provide investors with capital appreciation, as stock values typically rise over time, though they may fluctuate in the short term.
- Index: a type of mutual fund or exchange-traded fund (ETF) that aims to replicate the performance of a specific financial market index. In the context of an asset management company, index funds are a popular investment vehicle due to their simplicity, cost-effectiveness, and the ability to provide diversified exposure to a particular market or asset class.
- KPD: Kontrak Pengelolaan Dana (KPD) is the management of a securities portfolio for the benefit of a specific investor based on a bilateral and individual fund management agreement, structured in accordance with the regulations of the Financial Services Authority.
- Money Market: a type of mutual fund that invests in short-term, high-quality, low-risk debt instruments, such as Treasury bills, commercial paper, certificates of deposit, and other highly liquid, low-risk securities. The primary objective of a money market fund is to preserve capital and provide liquidity while offering a modest return, typically in the form of interest. In the context of an asset management company, money market funds are designed to provide a safe place to park cash for investors who are seeking minimal risk, immediate liquidity, and stable returns.
- Protected: A protected fund, also known as a capital-protected fund or guaranteed fund, is a type of investment fund designed to provide investors with a level of protection on their original capital while still offering the potential for growth. These funds aim to limit downside risk by ensuring that, at a minimum, the investor will get back a certain percentage of their initial investment, often at the end of a specified term.

definition of columns 'category' in data_input:
- AUM: AUM (Assets Under Management) refers to the total market value of all the financial assets that an asset management company or financial institution manages on behalf of its clients. It encompasses all the investments such as stocks, bonds, real estate, and other assets that the firm oversees and administers for both individual and institutional clients.
- Revenue: this represents the Gross Revenue coming from Management Fees in STAR AM based on each Product within each Clients. 
- Sharing Fee: A sharing fee in the context of an asset management company typically refers to a fee structure where the management firm shares a portion of the profits generated from investments with its clients or investors. This type of fee arrangement can align the interests of the asset manager with those of the clients, as it incentivizes the manager to maximize investment performance.
- OJK Fee: OJK Fee refers to fee charged by Indonesian Financial Services Authority. OJK Fees is charged based on the Gross Revenue generated from each Product and Client. OJK Fees charging may differ between each Product. 
- Net Revenue: represents the Net Revenue received by STAR AM. this is equivalent with Gross Profit which formula is (Revenue - Sharing Fee - OJK Fee). Sharing Fee and OJK Fee in STAR AM are treated as Direct Cost. 

Note that all value in 'mtd_value' and 'ytd_value' column nominal are displayed in IDR currency.
- mtd_value: refers to the numbers that display the cumulative balances from all products starting from the first date of the month up to the last day within that month. this field indicates month to date value. 
- ytd_value: refers to the numbers that display the cumulative balances from all products starting from the beginning of the fiscal year up to specific month. this field indicates year to date value. 

Here the steps for you to summarize and give insight about the STAR AM finance product segment data:
- Step 1 : Step by step analyze provided product segment data trends in mtd_value or ytd_value column as user request within years over months from each fund_type. Always show in numeric number instead in scientific number format. You don't need to use the tools for this step.
- Step 2 : Enhance your analysis from Step 1 by gathering additional insights from the internet to strengthen your summary. You are limited to a maximum of three internet searches. If you find the information you need before reaching three searches, you can proceed to the next step without completing all three searches. But if the question involves creating a chart, you can use the Chart Generator tool by passing the user question without paraphrasing for creating chart.
- Step 3 : Summarize the findings and provide insights based on your analysis from Step 1 and the additional information from Step 2. Ensure your final answer integrates the data trends with insights from the internet searches or chart generator.
- Step 4 : In the final output, You should include all reference data & links to back up your research; You should include all reference data.

If the question is a follow-up question or does not relate to the provided finance data, then here the steps for you:
- Step 1 : Get the information from the internet to get answer from the user question. REMEMBER YOU ARE ONLY PERMITTED TO SEARCH FROM THE INTERNET 3 TIMES OR LESS! If you feel enough with your research from the internet less than 3 times, you can immediately move on to the next step.
- Step 2 : From step 1, provide the final answer. In the final output, You should include all reference data & links to back up your research; You should include all reference data.
'''

suffix = '''Finance Product Segment Data of STAR AM Business Unit (if the data only contains 1 row of data, then it's most likely the answer to the user's question): 
```
{data_input}
```

Your past conversation with human:
```
{chat_history}
```

Begin!

Question: {human_input}
Thought: {agent_scratchpad}
'''

# Define the format instructions for the agent's output
format_instructions = """Strictly use the following format and it must be in consecutive order without any punctuation:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, MUST be one of these tool names only without the parameters [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated only 3 times)
Thought: I now know the final answer
Final Answer: If the question directly involves analyzing the finance product segment data provided. 

Provide your final answer using the following output format for each fund type:
<Fund Type>
- Summarization: <Your Summarization as a paragraph> 
- Insight: <Your Insight as a paragraph>


If the question is a follow-up or does not relate to the provided finance data:
Final Answer: <Directly provide the summarized answer without the detailed format>
"""

# Create the prompt for the ZeroShotAgent
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    format_instructions=format_instructions,
    input_variables=["data_input", "chat_history", "human_input", "agent_scratchpad"],

)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# stream_handler = StreamHandler(st.empty())

# Initialize the language model chain with a chat model
llm_chain = LLMChain(
    llm=llm,
    #     ChatOpenAI(
    #     model_name="gpt-4o",
    #     temperature=0
    #     # streaming=True,
    #     # callbacks=[stream_handler]
    # ),
    prompt=prompt
)

# Create the ZeroShotAgent with the language model chain
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

# Initialize the AgentExecutor with the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)

if __name__ == "__main__":

    # Setup the Streamlit interface
    st.title('Q&A AI Finance')

    st.subheader("STAR AM Product Segment", divider="orange")

    # Initialize session state for maintaining conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # if "example_query" not in st.session_state:
    #     st.session_state.example_query = None

    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "<img" in message['content']:
                image_path = re.search(r'src="([^"]*)"', message['content']).group(1)
                new_message = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', message['content'])
                st.markdown(new_message)
                st.image(image_path)
            else:
                st.markdown(message['content'])

    # User inputs their question
    user_question = st.chat_input("Enter your question about the finance data...")

    # Button to process the question
    if user_question:
        # Append user's question to the session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Run the agent with the formulated prompt
        with st.spinner('Processing...'):
            try:
                _, data_input = get_data(user_question=user_question)
                print(data_input)
                response = agent_executor.run(
                    data_input=data_input,
                    human_input=user_question
                )
            except:
                response = "I'm sorry I can't process your query right now. Please try again."

        # Append AI response to the session state
        try:
            with st.chat_message("assistant"):
                if "<img" in response:
                    image_path = re.search(r'src="([^"]*)"', response).group(1)
                    new_response = re.sub(r'<img src="[^"]*" alt="[^"]*">', '', response)
                    st.write_stream(stream_data(new_response))
                    st.image(image_path)
                else:
                    st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
        except:
            with st.chat_message("assistant"):
                response = "I'm sorry I can't process your query right now. Please try again."
                st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
