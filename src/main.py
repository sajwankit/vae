from load_config import ConfigLoader
from data_ingestion import DataIngestionPipeline

if __name__=="__main__":
    config = ConfigLoader(config_path="/Users/ankitsajwan/tech/projects/vae/config/config.yaml").config
    ingestion_pipeline = DataIngestionPipeline(config)
    pipeline.load_data()
    pipeline.aggregate_per_minute()
    pipeline.fill_missing_minutes()
    df = pipeline.get_aggregated_data()
    print(df.head())