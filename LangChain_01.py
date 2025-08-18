from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

#made a pipeline to use the model using transformer
pipe = pipeline("text-generation",
                model="gpt2",
                max_new_tokens=50
                );
def prec():
    #Just an Example how to use the LLm + prompt template
    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = PromptTemplate.from_template("Who is {celebrity}? ")
    chain = LLMChain(llm=llm, prompt=prompt)
    res = chain.run("Imran Khan")
    print(res)


#LLm to get name for you next SaaS app
prompt = PromptTemplate.from_template("What is the name of the Ecommerce Store that sells {Products}? ")
llm = HuggingFacePipeline(pipeline=pipe)
chain1 = LLMChain(llm=llm, prompt=prompt)


#LLm to get name for products from an ecommerce name
prompt = PromptTemplate.from_template("What are the names of the products at {store}? ")
llm = HuggingFacePipeline(pipeline=pipe)
chain2 = LLMChain(llm=llm, prompt=prompt)

#An Overall chain by combining both the chains:

chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

chain.run("candles")

