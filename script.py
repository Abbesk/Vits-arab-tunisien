from sklearn.model_selection import train_test_split
import datasets

# Charger le dataset
dataset = datasets.load_dataset("linagora/Tunisian_Derja_Dataset", "Derja_tunsi")

# Convertir l'ensemble 'train' en un DataFrame Pandas pour faciliter la division
train_df = dataset['train'].to_pandas()

# Diviser l'ensemble 'train' en train, validation, et test
train_data, temp_data = train_test_split(train_df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Afficher les tailles pour vérifier
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# Sauvegarder les sous-ensembles sous forme de fichiers CSV
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
