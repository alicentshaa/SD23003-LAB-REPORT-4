import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Activation Function Visualizer",
    layout="centered",
)

st.title("Neural Network Activation Functions")
st.write(
    """
    This application visualizes the activation functions:
    **Sigmoid**.
    """
)

x = np.linspace(-10, 10, 200)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots()
ax.plot(x, y, color="blue", label="Sigmoid")
ax.grid(True)
ax.set_title("Sigmoid Function")
st.pyplot(fig)

st.write("Sigmoid squashes values between 0 and 1.")