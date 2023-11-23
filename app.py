import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import tiktoken
from langchain.chains.summarize import load_summarize_chain
from time import monotonic
import textwrap


import streamlit as st
from newspaper import Article
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import tiktoken
from langchain.chains.summarize import load_summarize_chain
from time import monotonic
import textwrap


# Streamlit app
def main():
    st.title("Blog Summarizer")

    # API Key and Model Name
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
    model_name = "gpt-3.5-turbo"

    # Input for blog URL
    blog_url = st.text_input("Enter the URL of the blog post:")

    if st.button("Summarize"):
        if blog_url and OPENAI_API_KEY:
            try:
                article_text = extract_article_text(blog_url)
                summary = summarize_text(article_text, OPENAI_API_KEY, model_name)
                st.subheader("Markdown Summary")
                st.markdown(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error(
                "Please input both the URL of the blog post and the OpenAI API Key."
            )


def extract_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def summarize_text(news_article, OPENAI_API_KEY, model_name):
    # Splitting the text
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name=model_name)
    texts = text_splitter.split_text(news_article)
    docs = [Document(page_content=t) for t in texts]

    # Setting up the language model
    llm = ChatOpenAI(
        temperature=1, openai_api_key=OPENAI_API_KEY, model_name=model_name
    )

    # Prompt template
    prompt_template = """
    Imaging you are a master of story-telling and have been summarizing great blogs for years.
    Creating a summarized blog while maintaining a structured, intuitive, and simpler English touch involves focusing on key elements of the original content. Here's a suggested structure that combines storytelling and a conversational tone: Keep in mind while summarizing to Summarize the original blog by simplifying language, using a friendly and engaging tone. Begin with a captivating hook, flow conversationally, and maintain the author's voice. Break down complex ideas, sprinkle in humor or anecdotes if appropriate, and ask rhetorical questions to enhance reader engagement. Integrate visuals if available and emphasize the significance of key points. Be concise, read aloud for fluency, and conclude with a memorable statement. Review for accuracy to ensure the summary faithfully represents the original blog while delivering a pleasing, conversational, and intuitive experience.
    It's essential to maintain the tone and language of the original blog, carefully analyze the author's writing style, paying attention to their choice of words, sentence structure, and overall tone. Incorporate similar language, humor, or expressions to replicate the author's voice. Use direct quotes from the original blog when appropriate and be mindful of the overall mood and atmosphere created by the author. This approach helps preserve the unique identity of the content and ensures that readers experience a seamless transition between the original blog and its summarized version. Below are some instructions you have to strictly follow.

    IMPORTANT: ALWAYS USE FIRST PERSON LANGUAGE IN THE SUMMARY TO MAKE IT SOUND CONVERSATIONAL. USE HUMOUR AND CASUAL LANGUAGE IN BETWEEN THE SUMMARY TO MAKE IT INTERESTING, BUT DON'T DO IT LOT.
    IMPORTANT: ALWAYS START WITH: "Greeting! I am `author name` In this blog we'll be looking at:", this is just to give a gist of the what we are talking about, then continue the summary with first person voice in the next line.
    IMPORTANT: END THE BLOG WITH INTERESTING CONCLUSION. AND SAY "Be sure to read my/our full blog on" and then write the name of the blog site. MAKE SURE YOU DO NOT USE THE PHRASE "Conclusion", the word "CONCLUSION" is strictly prohibited.

    Manifest the personality of the author as yourself and give response like a human narator. End with an interesting conclusion.

    Here's the blog content:

{text}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Counting tokens
    num_tokens = num_tokens_from_string(news_article, model_name)

    # Load summarization chain
    gpt_35_turbo_max_tokens = 4097
    verbose = True
    chain = load_chain(num_tokens, gpt_35_turbo_max_tokens, llm, prompt, verbose)

    # Running the summarization chain
    start_time = monotonic()
    summary = chain.run(docs)
    run_time = monotonic() - start_time

    return summary


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_chain(num_tokens, model_max_tokens, llm, prompt, verbose):
    if num_tokens < model_max_tokens:
        return load_summarize_chain(
            llm, chain_type="stuff", prompt=prompt, verbose=verbose
        )
    else:
        return load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
