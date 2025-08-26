from models import MLModelTrainer

# Load your data
trainer = MLModelTrainer('data/ds1302.csv')
trainer.explore_data()

# Prepare data (replace 'your_target_column' with actual column name)
trainer.prepare_data('dead')
trainer.explore_data()

# Train all classification models with hyperparameter tuning
trainer.train_classification_models(use_grid_search=True)

# Compare performance
trainer.compare_models('classification')

# Get the best model
best_name, best_model = trainer.get_best_model('classification')