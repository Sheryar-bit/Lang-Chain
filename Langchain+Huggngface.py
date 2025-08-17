from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True
)

# A simple txt generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=pipe)

print("Explain quantum computing in simple terms")
print(llm.invoke("Explain quantum computing in simple terms"))


#Creating and testing a custom prompt template (apna banaya hua)
template = """You are an expert AI assistant. Provide detailed answers.
Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm  #First applying the prompt template then will pass the result to the LLM(this is the modern way, traditional as parameter based)

response = chain.invoke({"question": "How to scale your backend server upto 1 million users?"})
print("\nDetailed ans:")
print(response )

# another advanced usage could be like multi conversations
conversation = [
    {"role": "user", "content": "Tell me about the history of the Pakistan"},
    {"role": "assistant", "content": "Pakistan was..."},
    {"role": "user", "content": "What are Pakistan cricket team greatest achievemnts?"}
]
#formatting a little bit
formatted_conv = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

#this will invoke ourr LLm
print(llm.invoke(f"Continue this conversation:\n{formatted_conv}\nassistant:"))