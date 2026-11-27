# FIREGUARD ID v2 — Feature Quality Report

## 1. Extended Schema Candidate Features (0% Missingness)
| feature                     |   missing_count |      mean |      std |      min |    median |       max |
|:----------------------------|----------------:|----------:|---------:|---------:|----------:|----------:|
| hotspot_density_km2         |               0 |    0.1789 |   0.465  |    0     |    0.0325 |    7.0886 |
| fire_days                   |               0 |    8.5287 |  25.9752 |    0     |    3      |  472      |
| active_years_count          |               0 |    2.0164 |   1.7702 |    0     |    2      |    5      |
| frp_total_density_mw_km2    |               0 |    1.084  |   3.1761 |    0     |    0.1183 |   42.1954 |
| dem_glo90_mean_m            |               0 |   55.7722 | 100.849  |    0     |   19.9723 |  887.041  |
| dem_glo90_std_m             |               0 |   19.0915 |  35.4195 |    0     |    6.5727 |  274.842  |
| era5_dry_t2m_mean_c         |               0 |   26.3287 |   0.5137 |   23.74  |   26.3712 |   27.8346 |
| era5_dry_soil_moisture_mean |               0 |    0.3571 |   0.0627 |    0.035 |    0.3583 |    0.5432 |
| era5_annual_precip_mm       |               0 | 3057.47   | 205.219  | 2427.05  | 3059.85   | 3628.09   |
| mcd12q1_forest_share        |               0 |    0.489  |   0.3184 |    0     |    0.466  |    1      |
| mcd12q1_shrub_savanna_share |               0 |    0.3939 |   0.2916 |    0     |    0.4021 |    1      |

## 2. Pearson Correlation Matrix
|                             |   hotspot_density_km2 |   fire_days |   active_years_count |   frp_total_density_mw_km2 |   dem_glo90_mean_m |   dem_glo90_std_m |   era5_dry_t2m_mean_c |   era5_dry_soil_moisture_mean |   era5_annual_precip_mm |   mcd12q1_forest_share |   mcd12q1_shrub_savanna_share |
|:----------------------------|----------------------:|------------:|---------------------:|---------------------------:|-------------------:|------------------:|----------------------:|------------------------------:|------------------------:|-----------------------:|------------------------------:|
| hotspot_density_km2         |                 1     |       0.771 |                0.411 |                      0.813 |              0.002 |             0.032 |                 0.011 |                        -0.068 |                   0.071 |                 -0.193 |                         0.218 |
| fire_days                   |                 0.771 |       1     |                0.407 |                      0.376 |             -0.002 |             0.026 |                -0.032 |                         0.008 |                   0.079 |                 -0.139 |                         0.16  |
| active_years_count          |                 0.411 |       0.407 |                1     |                      0.335 |             -0     |             0.078 |                -0.117 |                         0.056 |                   0.158 |                 -0.317 |                         0.442 |
| frp_total_density_mw_km2    |                 0.813 |       0.376 |                0.335 |                      1     |              0.021 |             0.052 |                -0.01  |                        -0.044 |                   0.084 |                 -0.159 |                         0.196 |
| dem_glo90_mean_m            |                 0.002 |      -0.002 |               -0     |                      0.021 |              1     |             0.899 |                -0.775 |                         0.287 |                   0.061 |                  0.352 |                        -0.222 |
| dem_glo90_std_m             |                 0.032 |       0.026 |                0.078 |                      0.052 |              0.899 |             1     |                -0.743 |                         0.263 |                   0.02  |                  0.339 |                        -0.225 |
| era5_dry_t2m_mean_c         |                 0.011 |      -0.032 |               -0.117 |                     -0.01  |             -0.775 |            -0.743 |                 1     |                        -0.649 |                  -0.273 |                 -0.31  |                        -0.007 |
| era5_dry_soil_moisture_mean |                -0.068 |       0.008 |                0.056 |                     -0.044 |              0.287 |             0.263 |                -0.649 |                         1     |                   0.191 |                  0.097 |                         0.161 |
| era5_annual_precip_mm       |                 0.071 |       0.079 |                0.158 |                      0.084 |              0.061 |             0.02  |                -0.273 |                         0.191 |                   1     |                 -0.176 |                         0.322 |
| mcd12q1_forest_share        |                -0.193 |      -0.139 |               -0.317 |                     -0.159 |              0.352 |             0.339 |                -0.31  |                         0.097 |                  -0.176 |                  1     |                        -0.73  |
| mcd12q1_shrub_savanna_share |                 0.218 |       0.16  |                0.442 |                      0.196 |             -0.222 |            -0.225 |                -0.007 |                         0.161 |                   0.322 |                 -0.73  |                         1     |

## 3. Baseline vs Extended Schema Governance
- `config/features_baseline.yaml` (`v2.0.0-baseline`) remains the **authoritative primary clustering schema** (4 historical NOAA-20 fire-activity features).
- `config/features_extended.yaml` (`v2.1.0-extended`) curates 0%-missingness environmental covariates (GLO-90 DEM zonal stats, Open-Meteo `ecmwf_ifs` 0.75-deg IDW, MODIS MCD12Q1 land-cover shares) while isolating coastal-NaN layers (CHIRPS v3.0, MOD13Q1 NDVI), partial proxies (BIG Peatland, OSM), secondary S-NPP validation (`snpp_*`), and forbidden `SYNTHETIC` SRC-10 fields.
