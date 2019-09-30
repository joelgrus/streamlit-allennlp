import streamlit as st
from allennlp import pretrained
import matplotlib.pyplot as plt
import numpy as np

st.header("AllenNLP Demo")

# Load the pretrained BiDAF model for question answering.
# (It's big, don't do this over dial-up.)
predictor = pretrained.bidirectional_attention_flow_seo_2017()

# Create a text area to input the passage.
passage = st.text_area("passage", "The Matrix is a 1999 movie starring Keanu Reeves.")

# Create a text input to input the question.
question = st.text_input("question", "When did the Matrix come out?")

# Use the predictor to find the answer.
result = predictor.predict(question, passage)

# From the result, we want "best_span", "question_tokens", and "passage_tokens"
start, end = result["best_span"]
question_tokens = result["question_tokens"]
passage_tokens = result["passage_tokens"]

# We want to render the paragraph with the answer highlighted.
# We'll do that using `st.markdown`. In particular, for each token
# if it's part of the answer span we'll **bold** it. Otherwise we'll
# leave it as it.
mds = [f"**{token}**" if start <= i <= end else token
       for i, token in enumerate(passage_tokens)]

# And then we'll just concatenate them with spaces.
st.markdown(" ".join(mds))

# We'd also like to make a heatmap of the passage-question attention.
# We'll use plt.imshow() for that.
attention = result["passage_question_attention"]

fig, ax = plt.subplots()
plt.imshow(attention)

# Make sure to show every tick
ax.set_xticks(np.arange(len(question_tokens)))
ax.set_yticks(np.arange(len(passage_tokens)))

# Use the tokens as the labels
ax.set_xticklabels(question_tokens)
ax.set_yticklabels(passage_tokens)

# And add it to our output
st.pyplot()
