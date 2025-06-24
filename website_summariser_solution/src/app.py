import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
import os

# Set page config
st.set_page_config(page_title="Website Assistant", layout="centered")

# Load environment variables
load_dotenv(override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# System prompt
system_prompt = """
You are an assistant that analyzes the contents of a website.
Aim is to provide a straightforward, correct, and polite response to a question that a user might ask about the website and answer them based on the content.
Respond in markdown.
"""

# Function to extract website content
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def fetch_website_content(url):
    try:
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        text = soup.get_text()
        return {"title": title.strip(), "text": text.strip()}
    except Exception as e:
        return {"title": "Error", "text": f"Could not fetch website content: {str(e)}"}

def user_prompt_for(website, user_request):
    return (
        f"You are looking at a website titled: {website['title']}.\n\n"
        f"Please do as mentioned below using the website contents:\n\n"
        f"{user_request}\n\n"
        f"Website content:\n{website['text']}"
    )

# App layout
with st.container():
    st.title("🔍 Website Assistant")

    user_request = st.text_area("What do you want to do with the website?", placeholder="Summarize, extract key points, list FAQs, etc.")
    website_url = st.text_input("Enter the URL of the website")

    if st.button("Get Solution"):
        if not user_request or not website_url:
            st.warning("Please enter both a request and a website URL.")
        else:
            with st.spinner("Processing..."):
                website = fetch_website_content(website_url)
                prompt = user_prompt_for(website, user_request)

                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"Error while fetching AI response: {str(e)}")
