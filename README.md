# CompetitiveLEM

This repository contains the code accompanying the paper:

> **"Game-theoretic Analysis of Suppliers’ Pricing Power in Thermal-Electric Local Energy Markets"**

[![DOI](https://img.shields.io/badge/DOI-energy.2024.133591-blue)](https://doi.org/10.1016/j.energy.2024.133591)

This implementation enables reproduction of the results and offers tools to explore different market scenarios.


## 📄 Paper Abstract
The integration of renewable energy resources can be facilitated by local energy markets (LEMs) for multiple energy commodities. While multi-energy LEMs are designed to alleviate entry barriers for smaller players, the limited capacity and participant count within an LEM can cause imperfect competition. This study formulates a Nash-Cournot game to analyze the market power of local suppliers in a thermal-electric LEM. By comparing the prices and welfare outcomes of a perfectly competitive market with those of an oligopolistic market, we highlight the impact of the number and mix of market participants on the distribution of social welfare and market efficiency. Additionally, we introduce and assess two strategies aimed at mitigating the supplier’s market power. Based on simulations of a LEM with demand and installed capacity similar to that of a large German city, we show that LEM should not be assumed to be perfectly competitive, as the strategic behavior of supplier agents significantly affects the market outcome. Our analyzed mitigation strategies exhibit the potential to curtail suppliers’ market power, redistribute social welfare, and contribute to more balanced market dynamics.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GAMS
- Path solver license for GAMS

### Installation
Provide steps to clone the repository and install dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/CompetitiveLEM.git
cd CompetitiveLEM
```

Install required libraries

```
pip install -r requirements.txt
```

## 🛠️ Usage

Input Data: 
This repository allows scenario customization without any coding. Simply edit/create a new the provided Excel file on the Maps dir to set up your desired parameters and configurations.

Running the Model: 
Execute the model using the provided main.py script. Change the model and scenario name before running the model. Command will generate the respective .gdx result files on the RUNS dir. 

## 📜 Citating
If you use this code in your research, please cite our paper:

Title: Game-theoretic Analysis of Suppliers’ Pricing Power in Thermal-Electric Local Energy Markets

Authors: Julia Barbosa, Florian Döllinger, Florian Steinke

DOI: https://doi.org/10.1016/j.energy.2024.133591

You can also use the following BibTeX entry for citation:
```
@article{BARBOSA2024133591,
title = {Game-theoretic analysis of suppliers’ pricing power in thermal-electric local energy markets},
journal = {Energy},
pages = {133591},
year = {2024},
issn = {0360-5442},
doi = {https://doi.org/10.1016/j.energy.2024.133591},
url = {https://www.sciencedirect.com/science/article/pii/S0360544224033693},
author = {Julia Barbosa and Florian Döllinger and Florian Steinke},
keywords = {Local energy markets, Nash-cournot game, Complementarity modeling, Local heat markets, Market power},
}
```

## 🤝 Contributing

Contributions are welcome! If you’d like to contribute:

Fork the repository.
Create a new branch for your feature or bugfix.
Submit a pull request with a detailed description of your changes.
Feel free to submit issues for any bugs or enhancement requests!




