import pandas as pd

def load_paysim_data(file_path='../data/raw/PaySim_log.csv'):
    """Loads raw PaySim synthetic log data."""
    df = pd.read_csv(file_path)
    return df