# Prior probability of rain
P_rain = 0.3

# Likelihood of observing clouds given rain
P_clouds_given_rain = 0.8

# Marginal probability of observing clouds
P_clouds = 0.5

# Posterior probability of rain given clouds
P_rain_given_clouds = bayes_theorem(P_clouds_given_rain * P_rain, P_clouds)

print(f"Probability of rain given clouds: {P_rain_given_clouds:.2f}")