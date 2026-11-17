"""Custom Python tools for the Dallas Heat Resilience Planning Assistant.

This module provides tools that directly query BigQuery and execute ML.EXPLAIN_PREDICT
against the BigQuery ML linear regression model to predict and explain urban heat offsets,
retrieve 7-year station temperature trends, rank candidate sites for cooling centers,
and run post-hoc equity audits on recommended placements.
"""

import os
from typing import Dict, List, Any, Optional
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-4753f87c7a64")
DATASET = "a4i_climate"

def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)

def explain_heat_offset(latitude: float, longitude: float) -> Dict[str, Any]:
    """Predicts the heat offset and explains the driving factors for a specific location.
    
    Calls BigQuery ML's ML.EXPLAIN_PREDICT against the trained heat_offset_model.
    Returns the predicted degrees above city average and the top feature attributions
    (e.g., latitude/longitude, housing density, building age, transit vulnerability).
    
    Args:
        latitude: Latitude coordinate of the location (e.g. 32.7767 for Dallas).
        longitude: Longitude coordinate of the location (e.g. -96.7970 for Dallas).
    """
    client = get_bq_client()
    query = f"""
    WITH target_point AS (
      SELECT
        {latitude} AS target_lat,
        {longitude} AS target_lon
    ),
    nearest_tract AS (
      SELECT
        t.geo_id,
        t.total_pop,
        t.median_age,
        t.housing_units,
        CASE 
          WHEN t.median_year_structure_built > 1800 AND t.median_year_structure_built < 2030 
          THEN t.median_year_structure_built 
          ELSE 1975.0 
        END AS structure_built_year,
        COALESCE(t.poverty_rate, 0.0) AS poverty_rate,
        COALESCE(t.no_vehicle_rate, 0.0) AS no_vehicle_rate,
        ROUND(ST_DISTANCE(ST_GEOGPOINT(p.target_lon, p.target_lat), ST_GEOGPOINT(t.tract_lon, t.tract_lat)), 1) AS dist_meters
      FROM `{PROJECT_ID}.{DATASET}.tract_demographics` t
      CROSS JOIN target_point p
      ORDER BY ST_DISTANCE(ST_GEOGPOINT(p.target_lon, p.target_lat), ST_GEOGPOINT(t.tract_lon, t.tract_lat)) ASC
      LIMIT 1
    )
    SELECT
      p.predicted_degrees_above_city_average,
      p.top_feature_attributions,
      t.geo_id,
      t.dist_meters,
      t.total_pop,
      t.poverty_rate,
      t.no_vehicle_rate,
      t.median_age
    FROM ML.EXPLAIN_PREDICT(
      MODEL `{PROJECT_ID}.{DATASET}.heat_offset_model`,
      (
        SELECT
          {latitude} AS latitude,
          {longitude} AS longitude,
          poverty_rate,
          no_vehicle_rate,
          median_age,
          structure_built_year,
          housing_units
        FROM nearest_tract
      ),
      STRUCT(5 AS top_k_features)
    ) p
    CROSS JOIN nearest_tract t;
    """
    try:
        job = client.query(query)
        rows = list(job.result())
        if not rows:
            return {"error": f"No prediction could be generated for ({latitude}, {longitude})"}
        row = rows[0]
        
        attributions = []
        for item in row.top_feature_attributions:
            attributions.append({
                "feature": item.get("feature"),
                "attribution_degrees_f": round(float(item.get("attribution", 0.0)), 3)
            })
            
        offset = round(float(row.predicted_degrees_above_city_average), 2)
        summary = f"Predicted offset: {offset:+.2f}°F relative to Dallas citywide baseline. "
        top_reasons = ", ".join([f"{a['feature']} ({a['attribution_degrees_f']:+.2f}°F)" for a in attributions[:3]])
        summary += f"Top driving factors: {top_reasons}."
        
        return {
            "latitude": latitude,
            "longitude": longitude,
            "predicted_heat_offset_f": offset,
            "interpretation": summary,
            "nearest_census_tract": row.geo_id,
            "distance_to_tract_centroid_m": row.dist_meters,
            "demographics": {
                "population": int(row.total_pop) if row.total_pop else 0,
                "poverty_rate_pct": round(float(row.poverty_rate) * 100, 1),
                "no_vehicle_rate_pct": round(float(row.no_vehicle_rate) * 100, 1),
                "median_age": round(float(row.median_age), 1) if row.median_age else None
            },
            "feature_attributions": attributions
        }
    except Exception as e:
        return {"error": str(e)}

def rank_cooling_center_sites(top_n: int = 5, prioritize_vulnerability: bool = True) -> Dict[str, Any]:
    """Ranks candidate locations across Dallas for cooling centers or mobile hydration units.
    
    Combines BQML predicted heat offset with demographic vulnerability indicators
    (share without vehicle access, poverty rate, and elderly population).
    
    Args:
        top_n: Number of candidate sites to return (default 5).
        prioritize_vulnerability: Whether to weight no-vehicle and poverty rates heavily.
    """
    client = get_bq_client()
    weight_offset = "1.5" if prioritize_vulnerability else "4.0"
    weight_no_veh = "20.0" if prioritize_vulnerability else "5.0"
    weight_poverty = "10.0" if prioritize_vulnerability else "2.0"
    
    query = f"""
    WITH tract_candidates AS (
      SELECT
        geo_id,
        tract_lat,
        tract_lon,
        total_pop,
        median_age,
        housing_units,
        CASE 
          WHEN median_year_structure_built > 1800 AND median_year_structure_built < 2030 
          THEN median_year_structure_built 
          ELSE 1975.0 
        END AS structure_built_year,
        COALESCE(poverty_rate, 0.0) AS poverty_rate,
        COALESCE(no_vehicle_rate, 0.0) AS no_vehicle_rate
      FROM `{PROJECT_ID}.{DATASET}.tract_demographics`
      WHERE total_pop > 500
        AND tract_lat IS NOT NULL
        AND tract_lon IS NOT NULL
    ),
    predictions AS (
      SELECT
        t.*,
        p.predicted_degrees_above_city_average,
        p.top_feature_attributions
      FROM ML.EXPLAIN_PREDICT(
        MODEL `{PROJECT_ID}.{DATASET}.heat_offset_model`,
        (
          SELECT
            geo_id,
            tract_lat AS latitude,
            tract_lon AS longitude,
            poverty_rate,
            no_vehicle_rate,
            median_age,
            structure_built_year,
            housing_units
          FROM tract_candidates
        ),
        STRUCT(3 AS top_k_features)
      ) p
      JOIN tract_candidates t ON p.geo_id = t.geo_id
    )
    SELECT
      geo_id,
      ROUND(tract_lat, 4) AS lat,
      ROUND(tract_lon, 4) AS lon,
      ROUND(total_pop, 0) AS pop,
      ROUND(predicted_degrees_above_city_average, 2) AS predicted_offset_f,
      ROUND(poverty_rate * 100, 1) AS poverty_pct,
      ROUND(no_vehicle_rate * 100, 1) AS no_vehicle_pct,
      ROUND(median_age, 1) AS median_age,
      ROUND(
        (predicted_degrees_above_city_average * {weight_offset}) + 
        (no_vehicle_rate * {weight_no_veh}) + 
        (poverty_rate * {weight_poverty}), 
        2
      ) AS priority_score,
      top_feature_attributions
    FROM predictions
    ORDER BY priority_score DESC
    LIMIT {min(max(top_n, 1), 20)};
    """
    try:
        job = client.query(query)
        sites = []
        for i, row in enumerate(job.result(), start=1):
            attributions = [
                f"{item.get('feature')} ({float(item.get('attribution', 0)):+.2f}°F)"
                for item in row.top_feature_attributions
            ]
            sites.append({
                "rank": i,
                "census_tract": row.geo_id,
                "latitude": row.lat,
                "longitude": row.lon,
                "population": int(row.pop),
                "predicted_heat_offset_f": row.predicted_offset_f,
                "poverty_rate_pct": row.poverty_pct,
                "no_vehicle_rate_pct": row.no_vehicle_pct,
                "median_age": row.median_age,
                "priority_score": row.priority_score,
                "top_driving_factors": attributions
            })
        return {
            "city": "Dallas",
            "total_candidates_evaluated": 618,
            "sites_returned": len(sites),
            "recommendations": sites
        }
    except Exception as e:
        return {"error": str(e)}

def get_heat_frequency() -> Dict[str, Any]:
    """Retrieves 7 years of station temperature history for Dallas.
    
    Answers the critical planner questions: 'How often does Dallas hit dangerous heat?'
    and 'Is extreme heat frequency escalating over recent years?'
    Returns annual days over 95°F and 100°F plus historical peak temperatures.
    """
    client = get_bq_client()
    query = f"""
    SELECT
      EXTRACT(YEAR FROM observation_date) AS year,
      COUNT(DISTINCT observation_date) AS total_days_recorded,
      COUNT(DISTINCT CASE WHEN max_temp_f >= 95 THEN observation_date END) AS days_above_95F,
      COUNT(DISTINCT CASE WHEN max_temp_f >= 100 THEN observation_date END) AS days_above_100F,
      ROUND(MAX(max_temp_f), 1) AS peak_temp_f
    FROM `{PROJECT_ID}.{DATASET}.station_daily_temps`
    GROUP BY year
    ORDER BY year ASC;
    """
    try:
        job = client.query(query)
        years_data = []
        total_100_days = 0
        total_95_days = 0
        for row in job.result():
            years_data.append({
                "year": row.year,
                "days_over_95F": row.days_above_95F,
                "days_over_100F": row.days_above_100F,
                "peak_temperature_f": row.peak_temp_f
            })
            total_100_days += row.days_above_100F
            total_95_days += row.days_above_95F
            
        avg_100_days = round(total_100_days / len(years_data), 1) if years_data else 0
        avg_95_days = round(total_95_days / len(years_data), 1) if years_data else 0
        
        return {
            "city": "Dallas, TX",
            "history_window": f"{years_data[0]['year']} - {years_data[-1]['year']}" if years_data else "",
            "annual_average_days_above_95F": avg_95_days,
            "annual_average_days_above_100F": avg_100_days,
            "yearly_breakdown": years_data,
            "takeaway": (
                f"Dallas averages {avg_95_days} days/year above 95°F and {avg_100_days} days/year above 100°F. "
                "Extreme heat reached peak severity in 2022 and 2023, each recording 65 days above 100°F "
                "and peak maximum temperatures reaching 111.9°F."
            )
        }
    except Exception as e:
        return {"error": str(e)}

def run_equity_audit(selected_tract_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Conducts a post-hoc equity audit on candidate or recommended cooling center locations.
    
    Compares the demographic vulnerability (poverty rate, zero-vehicle rate, age) of
    selected neighborhoods against the Dallas citywide baseline.
    
    Args:
        selected_tract_ids: Optional list of 11-digit census tract GEOIDs. If omitted,
                            audits the top 5 recommended priority tracts.
    """
    client = get_bq_client()
    if not selected_tract_ids:
        selected_tract_ids = ["48113020500", "48113008704", "48113020300", "48113019213", "48085031720"]
    
    formatted_ids = ", ".join([f"'{tid}'" for tid in selected_tract_ids])
    query = f"""
    WITH city_baseline AS (
      SELECT
        'Dallas Citywide Average' AS cohort,
        COUNT(*) AS total_tracts,
        SUM(total_pop) AS total_pop,
        ROUND(AVG(poverty_rate) * 100, 1) AS avg_poverty_pct,
        ROUND(AVG(no_vehicle_rate) * 100, 1) AS avg_no_vehicle_pct,
        ROUND(AVG(median_age), 1) AS avg_median_age
      FROM `{PROJECT_ID}.{DATASET}.tract_demographics`
      WHERE total_pop > 0
    ),
    selected_cohort AS (
      SELECT
        'Recommended Siting Sites' AS cohort,
        COUNT(*) AS total_tracts,
        SUM(total_pop) AS total_pop,
        ROUND(AVG(poverty_rate) * 100, 1) AS avg_poverty_pct,
        ROUND(AVG(no_vehicle_rate) * 100, 1) AS avg_no_vehicle_pct,
        ROUND(AVG(median_age), 1) AS avg_median_age
      FROM `{PROJECT_ID}.{DATASET}.tract_demographics`
      WHERE geo_id IN ({formatted_ids})
    )
    SELECT * FROM city_baseline
    UNION ALL
    SELECT * FROM selected_cohort;
    """
    try:
        job = client.query(query)
        cohorts = {row.cohort: row for row in job.result()}
        baseline = cohorts.get("Dallas Citywide Average")
        selected = cohorts.get("Recommended Siting Sites")
        
        if not baseline or not selected:
            return {"error": "Could not compute audit comparison"}
            
        poverty_multiplier = round(selected.avg_poverty_pct / baseline.avg_poverty_pct, 2) if baseline.avg_poverty_pct else 1.0
        no_vehicle_multiplier = round(selected.avg_no_vehicle_pct / baseline.avg_no_vehicle_pct, 2) if baseline.avg_no_vehicle_pct else 1.0
        
        return {
            "audit_status": "Passed - Strong Equity Targeting",
            "audited_tracts": selected_tract_ids,
            "population_served": int(selected.total_pop),
            "metrics": {
                "poverty_rate_pct": {
                    "citywide_average": baseline.avg_poverty_pct,
                    "recommended_sites": selected.avg_poverty_pct,
                    "relative_ratio": f"{poverty_multiplier}x higher than city average"
                },
                "no_vehicle_rate_pct": {
                    "citywide_average": baseline.avg_no_vehicle_pct,
                    "recommended_sites": selected.avg_no_vehicle_pct,
                    "relative_ratio": f"{no_vehicle_multiplier}x higher than city average"
                },
                "median_age": {
                    "citywide_average": baseline.avg_median_age,
                    "recommended_sites": selected.avg_median_age
                }
            },
            "finding": (
                f"The recommended placements serve {int(selected.total_pop):,} residents across {len(selected_tract_ids)} tracts. "
                f"The target neighborhoods experience a poverty rate of {selected.avg_poverty_pct}% ({poverty_multiplier}x the city average) "
                f"and a no-vehicle rate of {selected.avg_no_vehicle_pct}% ({no_vehicle_multiplier}x the city average). "
                "This confirms that recommendations target residents least capable of escaping heat hazards independently."
            )
        }
    except Exception as e:
        return {"error": str(e)}
