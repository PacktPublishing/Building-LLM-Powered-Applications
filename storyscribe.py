import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper




st.set_page_config(page_title="StoryScribe", page_icon="📙")
st.header('📙 Welcome to StoryScribe, your story generator and promoter!')

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']

# Create a sidebar for user input
st.sidebar.title("Story teller and promoter")
st.sidebar.markdown("Please enter your details and preferences below:")

llm = OpenAI()

# Ask the user for age, gender and favourite movie genre
topic = st.sidebar.text_input("What is topic?", 'A dog running on the beach')
genre = st.sidebar.text_input("What is the genre?", 'Drama')
audience = st.sidebar.text_input("What is your audience?", 'Young adult')
social = st.sidebar.text_input("What is your social?", 'Instagram')

#story generator
story_template = """You are a storyteller. Given a topic, a genre and a target audience, you generate a story.

Topic: {topic}
Genre: {genre}
Audience: {audience}
Story: This is a story about the above topic, with the above genre and for the above audience:"""
story_prompt_template = PromptTemplate(input_variables=["topic", "genre", "audience"], template=story_template)
story_chain = LLMChain(llm=llm, prompt=story_prompt_template, output_key="story")

#post generator
social_template = """You are an influencer that, given a story, generate a social media post to promote the story.
The style should reflect the type of social media used.

Story: 
{story}
Social media: {social}
Review from a New York Times play critic of the above play:"""
social_prompt_template = PromptTemplate(input_variables=["story", "social"], template=social_template)
social_chain = LLMChain(llm=llm, prompt=social_prompt_template, output_key='post') 

#image generator

image_template = """Generate a detailed prompt to generate an image based on the following social media post:

Social media post:
{post}

The style of the image should be oil-painted.

"""

prompt = PromptTemplate(
    input_variables=["post"],
    template=image_template,
)
image_chain = LLMChain(llm=llm, prompt=prompt, output_key='image')

#overall chain

overall_chain = SequentialChain(input_variables = ['topic', 'genre', 'audience', 'social'], 
                chains=[story_chain, social_chain, image_chain],
                output_variables = ['story','post', 'image'], verbose=True)


if st.button('Create your post!'):
    result = overall_chain({'topic': topic,'genre':genre, 'audience': audience, 'social': social}, return_only_outputs=True)
    image_url = DallEAPIWrapper().run(result['image'])
    st.subheader('Story')
    st.write(result['story'])
    st.subheader('Social Media Post')
    st.write(result['post'])
    st.image(image_url)
