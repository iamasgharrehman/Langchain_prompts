from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate

# creating Chat History
chat_template = ChatPromptTemplate(
    [
        ('system', 'you are a helpful customer support assistant'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ]
)

# Loading the previous chat from database
chat_history=[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
print(chat_history)

prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query' : 'where is my refund'

})

print(prompt)
