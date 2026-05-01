# Reinforcement Learning for Dynamic Resource Allocation in 5G Networks

## Overview
This project implements a Q-Learning algorithm to solve the problem of dynamic 
resource allocation in 5G Ultra Reliable Low Latency Communications (URLLC) networks.

The core challenge: in URLLC networks, Statistical Channel State Information is 
unavailable at the transmitter due to strict latency constraints. This requires a 
dynamic, learning-based approach rather than conventional static methods.

## What This Project Does
- Models the resource allocation problem as a finite Markov Decision Process (MDP)
- Uses Instantaneous Channel State Information and packet queue lengths as inputs
- Implements Q-Learning to allocate channels to users based on learned experience
- Compares Q-Learning performance against a heuristic baseline algorithm
- Demonstrates that Q-Learning provides more flexibility while maintaining strong 
system performance

## Key Concepts
- Reinforcement Learning (Q-Learning)
- Markov Decision Process (MDP)
- 5G URLLC Networks
- Multi-user, multi-channel wireless resource allocation

## Tech Stack
- Python
- NumPy

## Results
Q-Learning outperformed the heuristic approach in flexibility while maintaining 
comparable overall system performance — demonstrating the viability of RL-based 
approaches for mission-critical wireless networks.

## Background
Developed as part of MSc Telecommunications Engineering dissertation at 
Loughborough University (2021) — awarded Distinction.
