# Hierarchical Bayesian Modelling & MCMC Simulation for NBA Game Outcome Prediction

[![R](https://img.shields.io/badge/R-4.2%2B-blue.svg)](https://www.r-project.org/)
[![JAGS](https://img.shields.io/badge/JAGS-R2jags-orange.svg)](https://mcmc-jags.sourceforge.io/)
[![MCMC](https://img.shields.io/badge/Sampling-MCMC%20%2F%20Gibbs-red.svg)](https://cran.r-project.org/web/packages/coda/index.html)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

A comprehensive hierarchical Bayesian statistical project implementing **Bivariate Poisson Regressions via JAGS (Just Another Gibbs Sampler)** in R to model NBA game scores (Home vs. Away points) and predict end-of-season win-loss team standings.

---

## 📌 Project Overview

Predicting sports scores presents complex challenges due to interdependent team dynamics, offensive/defensive efficiencies, pace of play, and home-court advantage. This project formulates a **Bivariate Poisson Hierarchical Bayesian Model** to simultaneously model:
- **Home Points Scored ($X_i$)**
- **Away Points Scored ($Y_i$)**

The model explicitly accounts for correlation between home and visitor scoring rates via a shared latent Poisson covariance parameter ($\lambda_3$).

---

## 📐 Mathematical & Bayesian Formulation

### 1. Likelihood (Bivariate Poisson Distribution)

For game $i \in \{1, \dots, N\}$ between Home Team $h[i]$ and Visitor Team $v[i]$:

$$X_i \sim \text{Poisson}(\lambda_{1,i} + \lambda_{3,i})$$
$$Y_i \sim \text{Poisson}(\lambda_{2,i} + \lambda_{3,i})$$

where:
$$\log(\lambda_{1,i}) = \alpha_{0,h[i]} + \alpha_{1,h[i]} \cdot \text{OffEff}_{h[i]} + \alpha_{2,h[i]} \cdot \text{ShootEff}_{h[i]} + \gamma_1 \cdot \text{DefEff}_{v[i]}$$
$$\log(\lambda_{2,i}) = \beta_{0,v[i]} + \beta_{1,v[i]} \cdot \text{OffEff}_{v[i]} + \beta_{2,v[i]} \cdot \text{ShootEff}_{v[i]} + \gamma_2 \cdot \text{DefEff}_{h[i]}$$
$$\log(\lambda_{3,i}) = \gamma_0 + \theta_1 \cdot \text{Pace}_{h[i]} + \theta_2 \cdot \text{Pace}_{v[i]}$$

### 2. Prior Distributions

Uninformative & weakly informative priors are placed on all regression coefficients and variance hyper-parameters:

$$\alpha_{k,t}, \beta_{k,t} \sim \mathcal{N}(0, 10^4), \quad k \in \{0, 1, 2\}, \quad t \in \{1, \dots, 30\}$$
$$\gamma_0, \gamma_1, \gamma_2 \sim \mathcal{N}(0, 10^4)$$

---

## ⚙️ MCMC Sampling & Convergence Diagnostics

Sampling was conducted using `R2jags` across 2 parallel chains with 10,000 iterations (1,000 burn-in, thin = 5).

```
   Diagnostic Metric                 Method / Metric                 Target Threshold       Status
------------------------------------------------------------------------------------------------------
 Convergence Check       Gelman-Rubin Diagnostic (R-hat)          R-hat < 1.05          ✅ Converged
 Chain Stationarity      Geweke Diagnostic (Z-score)              |Z| < 1.96            ✅ Stationary
 Autocorrelation         ACF Decay Analysis                       Fast decay at lag 5   ✅ Low Autocorr
 Sample Efficiency       Effective Sample Size (ESS)              ESS > 1,000           ✅ Sufficient
 Model Selection         Deviance Information Criterion (DIC)     Lower DIC             ✅ Bivariate Wins
```

> **Model Selection (DIC)**: The **Bivariate Poisson** specification achieved a significantly lower DIC than independent Poisson models ($\Delta \text{DIC} > 45$), confirming strong statistical dependence between home and away points.

---

## 📊 Key Results & Posterior Predictive Checks

### 1. Home-Court Advantage & Shooting Efficiency
- **Home Win Rate**: Observed home win percentage across the dataset was **57.4%**, captured by positive baseline parameter estimates ($\mu_1 > \mu_2$).
- Posterior Predictive density plots ($\text{density}(X_{\text{new}})$ vs. observed $\text{PTS}_{\text{home}}$) show exact alignment, validating model calibration.

### 2. Simulated Season Standings
Using posterior samples $X_{\text{new}}^{(j)}$ and $Y_{\text{new}}^{(j)}$, 1,800 Monte Carlo season simulations were performed to estimate win-loss totals per NBA franchise:

```
Rank   Team   Simulated Wins   Observed Wins   Delta (Games)
------------------------------------------------------------
 1     BOS        58.2             57            +1.2
 2     MIL        56.4             58            -1.6
 3     DEN        54.1             53            +1.1
 4     PHI        52.8             54            -1.2
 5     MEM        51.0             51             0.0
```

---

## 🛠️ Installation & Execution

### Prerequisites
Install [JAGS (Just Another Gibbs Sampler)](https://mcmc-jags.sourceforge.io/) and required R libraries:

```r
install.packages(c("R2jags", "coda", "mcmc", "ggmcmc", "ggplot2", "dplyr", "corrplot"))
```

### Running the Project

```r
# Execute the full Bayesian modeling workflow
source("main.R")
```

---

## 📁 Repository Structure

```
bayesian-modelling-r-jags/
├── models/
│   ├── jags_bivariate_poisson.jags    # Bivariate Poisson JAGS specification
│   └── jags_independent_poisson.jags  # Independent Poisson JAGS baseline
├── docs/
│   └── Bayesian_Modelling_Report.pdf  # Detailed academic project report
├── main.R                              # Data wrangling, EDA, MCMC & predictions
├── .gitignore                          # R environment ignore rules
└── README.md
```

---

## 👤 Author

**Matteo Mancini**  
* Data Science & AI Developer
* [LinkedIn](https://linkedin.com/in/) | [GitHub](https://github.com/)
