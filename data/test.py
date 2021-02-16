from data.process_data import load_data, clean_data

load = load_data('data/disaster_messages.csv', 'data/disaster_categories.csv')
df = clean_data(load)

