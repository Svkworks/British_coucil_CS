from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    lit,
    regexp_replace,
    col,
    asc_nulls_last,
    round,
    dense_rank,
    desc_nulls_last,
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from functools import reduce
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def initialize_spark():
    """
    Initialize a Spark session with the required configuration.

    Returns:
        SparkSession: A Spark session object.
    """
    try:
        spark = SparkSession.builder.appName("ICEYE").master("local[*]").getOrCreate()  # type: ignore
        spark.sparkContext.setLogLevel(
            "ERROR"
        )  # Set logging level to avoid verbose logs
        logging.info("Spark session initialized.")
        return spark
    except Exception as e:
        logging.error(f"Error initializing Spark session: {e}")
        raise e


def get_councils_df(spark: SparkSession, council_name: str) -> DataFrame:
    """
    Load a council dataset based on the council name.

    Args:
        spark (SparkSession): The active Spark session.
        council_name (str): Name of the council to load.

    Returns:
        DataFrame: A DataFrame containing the council data.
    """
    if council_name.startswith("property_"):
        path = f"assessment_data/{council_name}.csv"
    else:
        path = f"assessment_data/britain_councils/{council_name}.csv"
        return (
            spark.read.option("header", True)
            .csv(path)
            .withColumn("council_type", lit(council_name))
        )

    return spark.read.option("header", True).csv(path)


def process_all_councils(spark: SparkSession, council_names: list) -> DataFrame:
    """
    Load and process all council datasets, merging them into one DataFrame.

    Args:
        spark (SparkSession): The active Spark session.
        council_names (list): List of council names to load.

    Returns:
        DataFrame: A DataFrame containing all council data combined.
    """
    all_council_list = []
    for name in council_names:
        if name in ["property_avg_price", "property_sales_volume"]:
            # Store specific dataframes globally
            globals()[f"{name}_df"] = get_councils_df(spark, council_name=name)
        else:
            all_council_list.append(get_councils_df(spark, council_name=name))

    # Combine all council DataFrames into one
    all_council_df = reduce(DataFrame.unionAll, all_council_list)
    all_council_df = all_council_df.repartition("council")
    return all_council_df


def generate_avg_price_dataset(
    all_council_df: DataFrame, property_avg_price_df: DataFrame
):
    """
    Generate the top 10 authorities with the lowest change in average property prices from 2022 to 2023.

    Args:
        all_council_df (DataFrame): Combined council data.
        property_avg_price_df (DataFrame): Average property price data.

    Returns:
        DataFrame: A DataFrame containing the top 10 authorities.
    """
    avg_price_df = (
        (
            all_council_df.join(
                property_avg_price_df,
                property_avg_price_df["local_authority"] == all_council_df["council"],
                "left",
            ).withColumn(
                "difference",
                regexp_replace(col("difference"), "%", "").cast(DoubleType()),
            )
        )
        .orderBy(asc_nulls_last("difference"))
        .limit(10)
        .drop("local_authority")
    )

    avg_price_df.write.csv(
        "etl_output/council_avg_price.csv", header=True, sep=",", mode="overwrite"
    )
    logging.info("Average price dataset has been generated and saved.")
    return avg_price_df


def generate_sales_growth_dataset(
    all_council_df: DataFrame, property_sales_volume_df: DataFrame
):
    """
    Create a dataset ranking councils based on percentage growth in sales volume from 2022 to 2023.

    Args:
        all_council_df (DataFrame): Combined council data.
        property_sales_volume_df (DataFrame): Property sales volume data.

    Returns:
        DataFrame: A DataFrame containing the councils ranked by sales growth.
    """
    # Calculate sales volume growth percentage and round to 2 decimal places
    sales_growth_df = (
        all_council_df.join(
            property_sales_volume_df,
            property_sales_volume_df["local_authority"] == all_council_df["council"],
            "left",
        )
        .withColumn(
            "growth(%)",
            round(
                (
                    (col("sales_volume_nov_2023") - col("sales_volume_nov_2022"))
                    / col("sales_volume_nov_2022")
                    * 100
                ).cast("float"),
                2,
            ),
        )
        .drop("local_authority")
    )

    # Apply dense ranking to the growth percentages
    window = Window.orderBy(desc_nulls_last("growth(%)"))
    sales_growth_df = sales_growth_df.withColumn("rank", dense_rank().over(window))

    sales_growth_df.write.csv(
        "etl_output/colcil_sales_growth_df.csv", header=True, sep=",", mode="overwrite"
    )
    logging.info("Sales growth dataset has been generated and saved.")
    return sales_growth_df


def main():
    """
    Main function to run the ETL process for council average prices and sales growth analysis.
    """
    # Initialize Spark session
    spark = initialize_spark()

    # List of council datasets to load
    council_names = [
        "district_councils",
        "london_boroughs",
        "metropolitan_districts",
        "unitary_authorities",
        "property_avg_price",
        "property_sales_volume",
    ]

    # Process all council data and load specific datasets
    all_council_df = process_all_councils(spark, council_names)

    # Generate top 10 authorities based on average property price change
    avg_price_df = generate_avg_price_dataset(all_council_df, property_avg_price_df)  # type: ignore
    avg_price_df.show(10, truncate=False)

    # Generate council rankings based on sales growth
    sales_growth_df = generate_sales_growth_dataset(
        all_council_df, property_sales_volume_df  # type: ignore
    )
    sales_growth_df.show(10, truncate=False)


if __name__ == "__main__":
    main()
