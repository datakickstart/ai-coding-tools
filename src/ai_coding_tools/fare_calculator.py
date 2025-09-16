"""
Fare calculation utilities for taxi trip data.

This module provides functions to calculate fare per mile for taxi trips
using PySpark DataFrames.
"""

from typing import Optional
from pyspark.sql import DataFrame, functions as F


def add_fare_per_mile_column(
    df: DataFrame,
    fare_column: str = "fare_amount",
    distance_column: str = "trip_distance",
    output_column: str = "fare_per_mile"
) -> DataFrame:
    """
    Add a fare per mile column to a PySpark DataFrame.
    
    Calculates fare per mile by dividing fare amount by trip distance.
    Handles division by zero by setting null values for zero or negative distances.
    
    Args:
        df: Input PySpark DataFrame containing taxi trip data
        fare_column: Name of the column containing fare amounts (default: "fare_amount")
        distance_column: Name of the column containing trip distances (default: "trip_distance")
        output_column: Name for the new fare per mile column (default: "fare_per_mile")
        
    Returns:
        DataFrame with the new fare per mile column added
        
    Raises:
        ValueError: If required columns are missing from the DataFrame
        
    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("test").getOrCreate()
        >>> data = [(10.0, 2.0), (15.0, 3.0), (8.0, 0.0)]
        >>> df = spark.createDataFrame(data, ["fare_amount", "trip_distance"])
        >>> result_df = add_fare_per_mile_column(df)
        >>> result_df.show()
        +-----------+-------------+------------------+
        |fare_amount|trip_distance|     fare_per_mile|
        +-----------+-------------+------------------+
        |       10.0|          2.0|               5.0|
        |       15.0|          3.0|               5.0|
        |        8.0|          0.0|              null|
        +-----------+-------------+------------------+
    """
    # Validate input columns exist
    df_columns = df.columns
    missing_columns = []
    
    if fare_column not in df_columns:
        missing_columns.append(fare_column)
    if distance_column not in df_columns:
        missing_columns.append(distance_column)
        
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add fare per mile column with null handling for zero/negative distances
    result_df = df.withColumn(
        output_column,
        F.when(F.col(distance_column) > 0, F.col(fare_column) / F.col(distance_column))
        .otherwise(None)
    )
    
    return result_df


def calculate_fare_statistics(
    df: DataFrame,
    fare_per_mile_column: str = "fare_per_mile"
) -> dict:
    """
    Calculate basic statistics for fare per mile data.
    
    Args:
        df: DataFrame containing fare per mile data
        fare_per_mile_column: Name of the fare per mile column
        
    Returns:
        Dictionary containing min, max, avg, and count statistics
        
    Raises:
        ValueError: If the specified column doesn't exist
    """
    if fare_per_mile_column not in df.columns:
        raise ValueError(f"Column '{fare_per_mile_column}' not found in DataFrame")
    
    # Calculate statistics, excluding null values
    stats = df.filter(F.col(fare_per_mile_column).isNotNull()).agg(
        F.min(fare_per_mile_column).alias("min_fare_per_mile"),
        F.max(fare_per_mile_column).alias("max_fare_per_mile"),
        F.avg(fare_per_mile_column).alias("avg_fare_per_mile"),
        F.count(fare_per_mile_column).alias("count_non_null")
    ).collect()[0]
    
    return {
        "min_fare_per_mile": stats["min_fare_per_mile"],
        "max_fare_per_mile": stats["max_fare_per_mile"],
        "avg_fare_per_mile": stats["avg_fare_per_mile"],
        "count_non_null": stats["count_non_null"]
    }


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
