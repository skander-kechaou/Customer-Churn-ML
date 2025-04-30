import pandas as pd

# Assuming 'telcom' is your DataFrame containing the data
# Load the CSV file into a DataFrame
telcom = pd.read_csv("churn-bigml-20.csv")
unique_states = telcom['State'].unique()

print(unique_states)
