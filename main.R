# ==============================================================================
# HIERARCHICAL BAYESIAN MODELING FOR NBA GAME PREDICTION
# Model: Bivariate Poisson Regression via JAGS / R2jags
# Author: Matteo Mancini
# ==============================================================================

library(ggplot2)
library(dplyr)
library(R2jags)
library(coda)
library(corrplot)

cat("--- Step 1: Data Preparation & Feature Engineering ---\n")

# Team mapping dictionary
team_mapping <- c(
  "1610612740" = 1,  "1610612762" = 2,  "1610612739" = 3,  "1610612755" = 4,
  "1610612737" = 5,  "1610612738" = 6,  "1610612751" = 7,  "1610612752" = 8,
  "1610612745" = 9,  "1610612750" = 10, "1610612760" = 11, "1610612758" = 12,
  "1610612746" = 13, "1610612765" = 14, "1610612748" = 15, "1610612756" = 16,
  "1610612743" = 17, "1610612754" = 18, "1610612761" = 19, "1610612747" = 20,
  "1610612759" = 21, "1610612749" = 22, "1610612766" = 23, "1610612741" = 24,
  "1610612742" = 25, "1610612763" = 26, "1610612753" = 27, "1610612764" = 28,
  "1610612757" = 29, "1610612744" = 30
)

# Feature engineering helper
prepare_nba_data <- function(df) {
  df <- df %>%
    mutate(
      HOME_TEAM_ID = recode(HOME_TEAM_ID, !!!team_mapping),
      VISITOR_TEAM_ID = recode(VISITOR_TEAM_ID, !!!team_mapping)
    ) %>%
    filter(SEASON >= 2021)
  
  # Calculate possessions and paces
  df$tiri_da_2_home <- (df$PTS_home * 0.2) / (df$FG_PCT_home + 1e-5)
  df$tiri_da_2_away <- (df$PTS_away * 0.2) / (df$FG_PCT_away + 1e-5)
  df$tiri_da_3_home <- (df$PTS_home * 0.07) / (df$FG3_PCT_home + 1e-5)
  df$tiri_da_3_away <- (df$PTS_away * 0.07) / (df$FG3_PCT_away + 1e-5)
  df$tiri_liberi_home <- df$PTS_home * df$FT_PCT_home * 0.2
  df$tiri_liberi_away <- df$PTS_away * df$FT_PCT_away * 0.2
  
  df$Possessi_home <- df$tiri_da_2_home + df$tiri_da_3_home + df$tiri_liberi_home
  df$Possessi_away <- df$tiri_da_2_away + df$tiri_da_3_away + df$tiri_liberi_away
  
  df$paceH <- df$Possessi_home / 48
  df$paceA <- df$Possessi_away / 48
  
  df$off.eff_home <- df$PTS_home / (df$Possessi_home + 1e-5)
  df$off.eff_away <- df$PTS_away / (df$Possessi_away + 1e-5)
  df$shoot.eff_home <- df$FG_PCT_home + 0.5 * df$FG3_PCT_home + 0.44 * df$FT_PCT_home
  df$shoot.eff_away <- df$FG_PCT_away + 0.5 * df$FG3_PCT_away + 0.44 * df$FT_PCT_away
  df$def.eff_home <- df$PTS_away / (df$Possessi_away + 1e-5)
  df$def.eff_away <- df$PTS_home / (df$Possessi_home + 1e-5)
  
  return(df)
}

cat("--- Step 2: JAGS Bivariate Poisson Sampling ---\n")

run_jags_pipeline <- function(data) {
  jags_data <- list(
    N = nrow(data),
    nteams = 30,
    hometeam = data$HOME_TEAM_ID,
    awayteam = data$VISITOR_TEAM_ID,
    X = data$PTS_home,
    Y = data$PTS_away,
    off.eff_home = data$off.eff_home,
    off.eff_away = data$off.eff_away,
    shoot.eff_home = data$shoot.eff_home,
    shoot.eff_away = data$shoot.eff_away,
    def.eff_home = data$def.eff_home,
    def.eff_away = data$def.eff_away,
    pace_H = data$paceH,
    pace_A = data$paceA
  )
  
  params <- c("alpha0", "beta0", "gamma0", "gamma1", "gamma2", "Xnew", "Ynew")
  
  set.seed(123)
  fit <- jags(
    data = jags_data,
    parameters.to.save = params,
    model.file = "models/jags_bivariate_poisson.jags",
    n.chains = 2,
    n.iter = 10000,
    n.burnin = 1000,
    n.thin = 5,
    DIC = TRUE
  )
  
  cat("Model DIC:", fit$BUGSoutput$DIC, "\n")
  return(fit)
}

cat("Script execution ready. Call run_jags_pipeline(data) after loading dataset.\n")
