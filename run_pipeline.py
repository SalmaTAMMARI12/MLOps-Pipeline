from pipelines.training_pipeline import training_pipeline
import os

if __name__ == "__main__":
    DATA_PATH = "data/students_ai_usage.csv" 
    training_pipeline(
        data_path=DATA_PATH,
        model_type="logistic"
    )

    print("--- Pipeline terminée avec succès ! ---")
    print("Consulte MLflow UI pour voir les métriques et le modèle enregistré.")