from joblib import dump
from trained_models import trained_models_obj1, trained_models_obj2

# Save the entire list of models for Objective 1
dump(trained_models_obj1, 'insomnia_models.joblib')

# Save the entire list of models for Objective 2
dump(trained_models_obj2, 'anxiety_models.joblib')


