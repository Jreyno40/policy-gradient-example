# 🐍🛒 PyTorch Policy Gradient Cart-Pole Solver

Welcome to the PyTorch Policy Gradient Cart-Pole Solver repository! This repository contains two scripts that offer exciting functionalities.

## 📃 policygrad.py

- Trains a multi-layer perceptron as a policy model for the cartpole environment.
- Allows you to watch the model play a game in the cartpole OpenAI Gym environment by setting the `watch_play` flag.
- Plots the rewards over time when the `plot_rewards` flag is set.
- Invokes the `viz_conversion` script to store neurons, synapses, and weights for each episode in JSON format.

## 📃 viz_conversion.py

- Converts the recorded neurons, synapses, edges, and weights for each episode into a JSON document.
- Converts the rewards for each episode into a separate JSON document.
- Produces the `model.json` and `rewards.json` output files.

The generated `model.json` and `rewards.json` files can be utilized in my portfolio repository ([link](https://github.com/Jreyno40/jreyno40.github.io)) to create a captivating d3.js visualization.

Feel free to explore this repository and leverage the output files for a truly cool visualization experience!
