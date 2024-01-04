import streamlit as st
import numpy as np

# Function to simulate Markov Chain transitions
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    current_state = initial_state
    states = []
    for _ in range(num_steps):
        states.append(current_state)
        current_state = np.dot(current_state, transition_matrix)
    return states

# Streamlit app
st.title('Markov Chain Simulation')

# User input for transition matrix and initial state
transition_matrix_input = st.text_area('Enter Transition Matrix (rows separated by comma, each value separated by space):')
initial_state_input = st.text_input('Enter Initial State (values separated by space):')
num_steps_input = st.number_input('Enter Number of Steps:', min_value=1, max_value=1000, value=10)

if st.button('Simulate'):
    try:
        transition_matrix = np.array([list(map(float, row.split(','))) for row in transition_matrix_input.split('\n')])
        initial_state = np.array(list(map(float, initial_state_input.split())))
        
        # Normalize rows of transition matrix to ensure probabilities sum up to 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, None]
        
        # Simulate Markov Chain
        simulation_result = simulate_markov_chain(transition_matrix, initial_state, num_steps_input)
        
        st.write('Simulation Result:')
        st.write(np.array(simulation_result))
    except Exception as e:
        st.write('Please input valid data.')

