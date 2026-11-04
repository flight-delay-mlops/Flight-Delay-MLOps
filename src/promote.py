# Moves @production to the model in @staging, which passed the quality gate in train.py.
# Run by the Promote workflow, which then redeploys the API so it loads the new model.
from dotenv import load_dotenv
from mlflow import MlflowClient

from src.train import MODEL_NAME


def main():
    load_dotenv()  # MLflow address and login from .env (in CI they come from secrets)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, "staging").version
    client.set_registered_model_alias(MODEL_NAME, "production", version)
    print(f"@production is now version {version}")


if __name__ == "__main__":
    main()
