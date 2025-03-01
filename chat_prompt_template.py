from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

chat_template = ChatPromptTemplate(
    [
        ('system', 'You are a {domain} expert'),
        ('human','Explain in the simple terms, what is {topic}')
    ]
)

prompt = chat_template.invoke({
    'domain' : 'cricket', 'topic': 'dosra'
})

print(prompt)