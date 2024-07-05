from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question.

*IMPORTANT: DO NOT ANSWER IF YOU CANNOT FIND THE ANSWER IN CONTEXT. DO NOT USE YOUR OWN KNOWLEDGE*

*ONLY USE HISTORY IF THE QUESTION IS RELEVANT TO HISTORY*
"""


chat_template = """
 Instructions: You should answer the questions and queries asked by user.
 Give concise answer based on what is asked. Do not add extra information.

 NOTE:
 *DO NOT MAKE FURTHER QUESTIONS AND ANSWERS*
"""



def get_prompt_template(system_prompt=system_prompt,passed_prompt = None, promptTemplate_type='mistral', history=True):
    print('passed prompt in get prompt template is,',passed_prompt)
    if promptTemplate_type == "mistral":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt +('\n*IMPORTANT INSTRUCTION BELOW*\n' +passed_prompt + "\n" if passed_prompt is not None else "")+ E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""
            
            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            # print('prompt template is',prompt_template)
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    elif promptTemplate_type == "llama":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
               
                +
                ('\n*IMPORTANT INSTRUCTION BELOW*\n' +passed_prompt + "\n" if passed_prompt is not None else "")
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt

                +
                ('\n*IMPORTANT INSTRUCTION BELOW*\n' +passed_prompt + "\n" if passed_prompt is not None else "")
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferWindowMemory(input_key="question", memory_key="history",k=1,return_messages=True)

    return (
        prompt,
        memory,
    )
