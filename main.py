#%% [markdown]
# # Monte Carlo Simulation of Vehicle Component Failure
# 
# This notebook simulates and analyzes vehicle component failures using Monte Carlo methods.

#%% [markdown]
# ## Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import make_motortime_shorter

#%% [markdown]
# ## Set random seed for reproducibility
np.random.seed(42)
n_trials = 10000

#%% [markdown]
# ## Create empty DataFrame for storing results
columns = ['motor', 'sensor', 'control', 'vehicle_failure']
df = pd.DataFrame(index=range(n_trials), columns=columns)
print("Empty DataFrame (preview):")
print(df.head())

#%% [markdown]
# ## Run Monte Carlo simulation
for trial in range(n_trials):
    # Generate failure times for components
    motor_time = np.random.poisson(7000)
    sensor_time = np.random.normal(5000, 100)
    control_time = np.random.uniform(4000, 8000)

    # Apply condition: sensor failure reduces motor time, but only if the reduced time
    # would not be less than the sensor time
    # wir haben genug zeit
    motor_time = make_motortime_shorter(sensor_time, motor_time)

    # Calculate vehicle failure time (min of motor or control unit)
    vehicle_failure = min(motor_time, control_time)

    # Store results in DataFrame
    df.loc[trial, 'motor'] = motor_time
    df.loc[trial, 'sensor'] = sensor_time
    df.loc[trial, 'control'] = control_time
    df.loc[trial, 'vehicle_failure'] = vehicle_failure

print("DataFrame with simulated failure times (preview):")
print(df.head())

#%% [markdown]
# 4. Stelle die Zeit bis zum Ausfall in einem Histogramm dar (@fig-monte-carlo-failure-histogram).
# ## Plot histogram of vehicle failure times
plt.figure(figsize=(10, 6))
plt.hist(df['vehicle_failure'], bins=50, density=True, color='skyblue', alpha=0.7)
plt.xlabel('Vehicle Failure Time (hours)')
plt.ylabel('Density')
plt.title('Distribution of Vehicle Failure Times')
plt.grid(True, alpha=0.3)
plt.show()

#%% Plot stacked histogram of component failures
plt.figure(figsize=(10, 6))
plt.hist([df['motor'], df['sensor'], df['control']], 
         bins=50, 
         stacked=True,
         label=['Motor', 'Sensor', 'Control'],
         alpha=0.7)
plt.xlabel('Failure Time (hours)')
plt.ylabel('Count')
plt.title('Stacked Histogram of Component Failure Times')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%% [markdown]
# 5. Berechne für jede Komponente die Wahrscheinlichkeit, dass sie ausgefallen ist, gegeben das Fahrzeug fällt vor 5000 Stunden aus. 
# ## Analysis of failures before 5000 hours
early_failures = df[df['vehicle_failure'] < 5000]
total_early_failures = len(early_failures)

print(total_early_failures)
# Calculate probabilities for each component
probabilities = {
    'motor': len(early_failures[early_failures['motor'] < 5000]) / total_early_failures,
    'sensor': len(early_failures[early_failures['sensor'] < 5000]) / total_early_failures,
    'control': len(early_failures[early_failures['control'] < 5000]) / total_early_failures
}

print("\nProbabilities of component failure given vehicle failure before 5000 hours:")
for component, prob in probabilities.items():
    print(f"{component}: {prob:.4f}")

#%% [markdown]
# 6. Gibt es eine Korrelation zwischen allen Zeiten bis zum Ausfall  aller Komponenten und des Fahrzeugs? Berechne die Korrelationskoeffizienten und visualisiere die Korrelation in einer Scatterplot-Matrix.
# ## Correlation analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Component Failure Times')
plt.show()

#%% Scatterplot Matrix
plt.figure(figsize=(12, 12))
sns.pairplot(df[['motor', 'sensor', 'control', 'vehicle_failure']], diag_kind='hist')
plt.suptitle('Scatterplot Matrix of Component and Vehicle Failure Times', y=1.02)
plt.show()

#%% [markdown]
# 7. Wie hoch ist die Bedingte Wahrscheinlichkeit, dass die Steuereinheit den ausgefallen ist, wenn das Fahrzeug vor 4000 Stunden ausgefallen ist.
# ## Conditional probability: Control unit failure before 4000 hours

try:
    very_early_failures = df[df['vehicle_failure'] < 4000]
    control_failures = very_early_failures[very_early_failures['control'] < 4000]
    prob_control = len(control_failures) / len(very_early_failures)

    print(f"\nConditional probability of control unit failure given vehicle failure before 4000 hours: {prob_control:.4f}")
except ZeroDivisionError:
    print("Es gibt keine so frühen Ausfälle!")
          
#%% [markdown]
# 8. Wähle eine Verteilung für die Ausfallzeit des Fahrzeugs fitte diese. Vergleiche die Verteilung mit dem Histogramm und gibt die Parameter der Verteilung an.
# ## Fit distribution to vehicle failure times
# We'll fit a normal distribution to the vehicle failure times

# Convert data to numpy array and ensure it's numeric
failure_times = df['vehicle_failure'].to_numpy(dtype=float)

# Fit normal distribution
mean, std = stats.norm.fit(failure_times)

# Plot fitted distribution against histogram
plt.figure(figsize=(10, 6))
plt.hist(failure_times, bins=50, density=True, color='skyblue', alpha=0.7, label='Histogram')
x = np.linspace(min(failure_times), max(failure_times), 100)
plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2, label='Fitted Normal')
plt.xlabel('Vehicle Failure Time (hours)')
plt.ylabel('Density')
plt.title('Vehicle Failure Times with Fitted Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nFitted Normal distribution parameters:")
print(f"Mean: {mean:.4f}")
print(f"Standard deviation: {std:.4f}")

#%% [markdown]
# ## Calculate probability of failure before 3000 hours
prob_before_3000 = stats.norm.cdf(3000, mean, std)
print(f"\nProbability of vehicle failure before 3000 hours: {prob_before_3000:.4f}") 

#%% [markdown]
# ## Alternative approach using Kernel Density Estimation
# Instead of assuming a specific distribution, we can use kernel density estimation
# to get a non-parametric estimate of the distribution

# Create kernel density estimator
kde = stats.gaussian_kde(failure_times)

# Plot histogram with KDE
plt.figure(figsize=(10, 6))
plt.hist(failure_times, bins=50, density=True, color='skyblue', alpha=0.7, label='Histogram')
x = np.linspace(min(failure_times), max(failure_times), 100)
plt.plot(x, kde(x), 'r-', lw=2, label='Kernel Density Estimate')
plt.xlabel('Vehicle Failure Time (hours)')
plt.ylabel('Density')
plt.title('Vehicle Failure Times with Kernel Density Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate probability of failure before 3000 hours using KDE
# We need to integrate the KDE from 0 to 3000
prob_before_3000_kde = kde.integrate_box_1d(0, 3000)
print(f"\nProbability of vehicle failure before 3000 hours (KDE method): {prob_before_3000_kde:.4f}")

# Compare with normal distribution result
print(f"Probability of vehicle failure before 3000 hours (Normal distribution): {prob_before_3000:.4f}")

# %%
