"""
Fare calculation utilities for taxi trip data.

This module provides functions to calculate fare per mile for taxi trips
using PySpark DataFrames.
"""

from typing import Optional
from pyspark.sql import DataFrame, functions as F


def add_avg_fare_per_mile_by_zones(
    df: DataFrame,
    pickup_zone_column: str = "pickup_zip",
    dropoff_zone_column: str = "dropoff_zip",
    fare_column: str = "fare_amount",
    distance_column: str = "trip_distance",
    output_column: str = "avg_fare_per_mile_by_zones"
) -> DataFrame:
    """
    Add average fare per mile column based on pickup and destination zones.
    
    This function calculates the average fare per mile for each unique combination
    of pickup and dropoff zones, then joins this information back to the original
    DataFrame. This provides route-specific fare per mile averages.
    
    Args:
        df: Input PySpark DataFrame containing taxi trip data
        pickup_zone_column: Name of pickup zone column (default: "pickup_zip")
        dropoff_zone_column: Name of dropoff zone column (default: "dropoff_zip")
        fare_column: Name of fare amount column (default: "fare_amount")
        distance_column: Name of trip distance column (default: "trip_distance")
        output_column: Name for new average column (default: "avg_fare_per_mile_by_zones")
        
    Returns:
        DataFrame with the new average fare per mile by zones column added
        
    Raises:
        ValueError: If required columns are missing from the DataFrame
        
    Example:
        >>> df_with_avg = add_avg_fare_per_mile_by_zones(taxi_df)
        >>> df_with_avg.select("pickup_zip", "dropoff_zip", "avg_fare_per_mile_by_zones").show()
        +----------+-----------+-------------------------+
        |pickup_zip|dropoff_zip|avg_fare_per_mile_by_zones|
        +----------+-----------+-------------------------+
        |     10001|      10002|                     5.25|
        |     10001|      10003|                     4.80|
        +----------+-----------+-------------------------+
    """
    # Validate input columns exist
    df_columns = df.columns
    missing_columns = []
    
    required_columns = [pickup_zone_column, dropoff_zone_column, fare_column, distance_column]
    for col in required_columns:
        if col not in df_columns:
            missing_columns.append(col)
            
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # First, calculate individual fare per mile for each trip (excluding zero/negative distances)
    df_with_individual_fare_per_mile = df.withColumn(
        "temp_fare_per_mile",
        F.when(F.col(distance_column) > 0, F.col(fare_column) / F.col(distance_column))
        .otherwise(None)
    )
    
    # Calculate average fare per mile by pickup/dropoff zone combination
    zone_averages = df_with_individual_fare_per_mile.filter(
        F.col("temp_fare_per_mile").isNotNull()
    ).groupBy(
        pickup_zone_column, dropoff_zone_column
    ).agg(
        F.avg("temp_fare_per_mile").alias("zone_avg_fare_per_mile"),
        F.count("temp_fare_per_mile").alias("trip_count")
    )
    
    # Join the zone averages back to the original DataFrame
    result_df = df.join(
        zone_averages,
        on=[pickup_zone_column, dropoff_zone_column],
        how="left"
    ).withColumn(
        output_column,
        F.col("zone_avg_fare_per_mile")
    ).drop("zone_avg_fare_per_mile", "trip_count")
    
    return result_df
