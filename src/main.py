import mlflow


if __name__ == '__main__':
    
    with mlflow.start_run(run_name='main') as run:
        mlflow.run('.', 'stage01_get_data', use_conda=False)
        mlflow.run('.', 'stage02_train_model', use_conda=False)
        







