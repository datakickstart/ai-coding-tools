"""
Unit tests for fare_calculator module.

Tests the fare per mile calculation functionality using pytest and PySpark.
"""

import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

from ai_coding_tools.fare_calculator import (
    add_fare_per_mile_column, 
    calculate_fare_statistics, 
    add_avg_fare_per_mile_by_zones
)


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .appName("test_fare_calculator") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()


@pytest.fixture
def sample_taxi_data(spark):
    """Create sample taxi data for testing."""
    schema = StructType([
        StructField("fare_amount", DoubleType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_zip", StringType(), True),
        StructField("dropoff_zip", StringType(), True)
    ])
    
    data = [
        (10.0, 2.0, "10001", "10002"),  # fare_per_mile = 5.0
        (15.0, 3.0, "10003", "10004"),  # fare_per_mile = 5.0
        (8.0, 1.6, "10005", "10006"),   # fare_per_mile = 5.0
        (12.0, 0.0, "10007", "10008"),  # fare_per_mile = null (zero distance)
        (20.0, -1.0, "10009", "10010"), # fare_per_mile = null (negative distance)
        (0.0, 2.0, "10011", "10012"),   # fare_per_mile = 0.0
    ]
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def zone_taxi_data(spark):
    """Create sample taxi data with repeated zone combinations for testing averages."""
    schema = StructType([
        StructField("fare_amount", DoubleType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("pickup_zip", StringType(), True),
        StructField("dropoff_zip", StringType(), True)
    ])
    
    data = [
        # Zone A to B: avg = (4.0 + 6.0) / 2 = 5.0
        (8.0, 2.0, "10001", "10002"),   # fare_per_mile = 4.0
        (12.0, 2.0, "10001", "10002"),  # fare_per_mile = 6.0
        
        # Zone A to C: avg = 7.5
        (15.0, 2.0, "10001", "10003"),  # fare_per_mile = 7.5
        
        # Zone B to C: avg = (5.0 + 5.0) / 2 = 5.0  
        (10.0, 2.0, "10002", "10003"),  # fare_per_mile = 5.0
        (10.0, 2.0, "10002", "10003"),  # fare_per_mile = 5.0
        
        # Zero distance trip (should be excluded from averages)
        (20.0, 0.0, "10001", "10002"),  # fare_per_mile = null
        
        # New trip for existing zone combination
        (16.0, 2.0, "10001", "10002"),  # fare_per_mile = 8.0, new avg = (4+6+8)/3 = 6.0
    ]
    
    return spark.createDataFrame(data, schema)


class TestAddFarePerMileColumn:
    """Test cases for add_fare_per_mile_column function."""
    
    def test_valid_scenarios(self, sample_taxi_data):
        """Test function with various valid inputs including edge cases."""
        result_df = add_fare_per_mile_column(sample_taxi_data)
        
        # Test column addition
        assert "fare_per_mile" in result_df.columns
        
        # Test return type
        assert hasattr(result_df, 'collect')  # Is a DataFrame
        
        # Test column preservation  
        assert len(result_df.columns) == len(sample_taxi_data.columns) + 1
        
        # Test calculations with various scenarios
        results = result_df.collect()
        assert results[0]["fare_per_mile"] == 5.0   # 10.0 / 2.0
        assert results[1]["fare_per_mile"] == 5.0   # 15.0 / 3.0  
        assert results[2]["fare_per_mile"] == 5.0   # 8.0 / 1.6
        assert results[3]["fare_per_mile"] is None  # 12.0 / 0.0 -> null
        assert results[4]["fare_per_mile"] is None  # 20.0 / -1.0 -> null
        assert results[5]["fare_per_mile"] == 0.0   # 0.0 / 2.0
        
        # Test custom column names
        custom_result = add_fare_per_mile_column(
            sample_taxi_data.withColumnRenamed("fare_amount", "total_fare")
                           .withColumnRenamed("trip_distance", "distance_miles"),
            fare_column="total_fare",
            distance_column="distance_miles", 
            output_column="custom_fare_per_mile"
        )
        assert "custom_fare_per_mile" in custom_result.columns

    def test_error_handling(self, sample_taxi_data):
        """Test function error handling and edge cases."""
        # Test missing fare column
        df_no_fare = sample_taxi_data.drop("fare_amount")
        with pytest.raises(ValueError, match="Missing required columns: \\['fare_amount'\\]"):
            add_fare_per_mile_column(df_no_fare)
        
        # Test missing distance column
        df_no_distance = sample_taxi_data.drop("trip_distance")
        with pytest.raises(ValueError, match="Missing required columns: \\['trip_distance'\\]"):
            add_fare_per_mile_column(df_no_distance)
        
        # Test missing both columns
        df_minimal = sample_taxi_data.select("pickup_zip", "dropoff_zip")
        with pytest.raises(ValueError, match="Missing required columns: \\['fare_amount', 'trip_distance'\\]"):
            add_fare_per_mile_column(df_minimal)


class TestCalculateFareStatistics:
    """Test cases for calculate_fare_statistics function."""
    
    def test_valid_scenarios(self, sample_taxi_data):
        """Test statistics calculation with valid inputs."""
        df_with_fare_per_mile = add_fare_per_mile_column(sample_taxi_data)
        stats = calculate_fare_statistics(df_with_fare_per_mile)
        
        # Expected values: [5.0, 5.0, 5.0, null, null, 0.0] -> non-null: [5.0, 5.0, 5.0, 0.0]
        expected_stats = {
            "min_fare_per_mile": 0.0,
            "max_fare_per_mile": 5.0, 
            "avg_fare_per_mile": 3.75,  # (5.0 + 5.0 + 5.0 + 0.0) / 4
            "count_non_null": 4
        }
        assert stats == expected_stats
        
        # Test custom column name
        custom_df = add_fare_per_mile_column(sample_taxi_data, output_column="custom_fare_per_mile")
        custom_stats = calculate_fare_statistics(custom_df, "custom_fare_per_mile")
        assert custom_stats["count_non_null"] == 4

    def test_error_handling(self, sample_taxi_data):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError, match="Column 'fare_per_mile' not found in DataFrame"):
            calculate_fare_statistics(sample_taxi_data)


class TestAddAvgFarePerMileByZones:
    """Test cases for add_avg_fare_per_mile_by_zones function."""
    
    def test_valid_scenarios(self, zone_taxi_data):
        """Test zone-based average calculation with valid inputs."""
        result_df = add_avg_fare_per_mile_by_zones(zone_taxi_data)
        
        # Test column addition
        assert "avg_fare_per_mile_by_zones" in result_df.columns
        
        # Test return type
        assert hasattr(result_df, 'collect')  # Is a DataFrame
        
        # Test column preservation (no temp columns left)
        original_cols = set(zone_taxi_data.columns)
        result_cols = set(result_df.columns)
        assert original_cols.issubset(result_cols)
        assert len(result_df.columns) == len(zone_taxi_data.columns) + 1
        
        # Test calculations - collect and organize by route
        results = result_df.collect()
        route_averages = {}
        for row in results:
            route = (row["pickup_zip"], row["dropoff_zip"])
            avg = row["avg_fare_per_mile_by_zones"]
            if route not in route_averages and avg is not None:
                route_averages[route] = avg
        
        # Verify expected averages
        # Route 10001->10002: (4.0 + 6.0 + 8.0) / 3 = 6.0 (excluding zero distance)
        assert abs(route_averages[("10001", "10002")] - 6.0) < 0.001
        # Route 10001->10003: 7.5  
        assert abs(route_averages[("10001", "10003")] - 7.5) < 0.001
        # Route 10002->10003: 5.0
        assert abs(route_averages[("10002", "10003")] - 5.0) < 0.001
        
        # Test custom column names
        custom_result = add_avg_fare_per_mile_by_zones(
            zone_taxi_data.withColumnRenamed("pickup_zip", "origin")
                          .withColumnRenamed("dropoff_zip", "destination"),
            pickup_zone_column="origin",
            dropoff_zone_column="destination",
            output_column="custom_avg"
        )
        assert "custom_avg" in custom_result.columns

    def test_error_handling(self, zone_taxi_data, spark):
        """Test error handling and edge cases."""
        # Test missing required columns
        incomplete_df = zone_taxi_data.drop("pickup_zip")
        with pytest.raises(ValueError, match="Missing required columns: \\['pickup_zip'\\]"):
            add_avg_fare_per_mile_by_zones(incomplete_df)
        
        # Test with multiple missing columns
        minimal_df = zone_taxi_data.select("pickup_zip", "dropoff_zip")
        with pytest.raises(ValueError, match="Missing required columns: \\['fare_amount', 'trip_distance'\\]"):
            add_avg_fare_per_mile_by_zones(minimal_df)
        
        # Test empty DataFrame handling
        empty_schema = zone_taxi_data.schema
        empty_df = spark.createDataFrame([], empty_schema)
        empty_result = add_avg_fare_per_mile_by_zones(empty_df)
        assert empty_result.count() == 0
        assert "avg_fare_per_mile_by_zones" in empty_result.columns


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_end_to_end_workflow(self, zone_taxi_data):
        """Test complete workflow from raw data to zone-based averages."""
        # Add individual fare per mile
        individual_df = add_fare_per_mile_column(zone_taxi_data)
        
        # Add zone-based averages
        final_df = add_avg_fare_per_mile_by_zones(zone_taxi_data)
        
        # Verify workflow results
        assert final_df.count() == zone_taxi_data.count()
        assert "avg_fare_per_mile_by_zones" in final_df.columns
        
        # Test that both functions can work together
        combined_df = add_avg_fare_per_mile_by_zones(individual_df)
        assert "fare_per_mile" in combined_df.columns
        assert "avg_fare_per_mile_by_zones" in combined_df.columns
        assert combined_df.count() == zone_taxi_data.count()
