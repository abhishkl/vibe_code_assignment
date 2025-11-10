# “Vibe Code” POC - Clustered Forecast with Model Switching
**Objective:** Train 2 models and implement a distribution-shift trigger.

**Output:** 90-day forecast and model switches and a log of model switches (if any)

## Methodology

1. Generatebookingdemandacross3car-classclusters(Economy,SUV,Luxury).
2. Generate external travel signals such as HotelADR, AirlineADR, and Airline Cancellations.
3. Select 2 model families:Linear Regression and Random Forest Regressor
4. Uses Population Stability Index(PSI)to monitor data drift in key features.
5. When PSI exceeds a threshold(e.g.,2.5),the system switches model families for that cluster.

## Outputs

1. Processing of each cluster i.e. PSI calculation and the downstream decision based on the threshold.
2. Sample Forecast of a cluster.
3. Model Switch Log – providing a clear table of the model switch from old modelto new and the metric (PSI Score) on which it was based on.
4. Demand Forecast curves(90days)for each cluster.
5. Demand Forecast in forecast_results_PSI.csvfile.
6. Switch logs in model_switch_log_PSI.csvfile

**Note:** The POC is to validate how the Model switching works and the model families selected are just for testing. The PSI threshold of 2.5 is not at all in line with the real world thresholds which should not be more than 0.25 (which means significant change in distribution)

## How to Run the code?

1. Open Terminal on Mac or the“clustered_forecast_with_psi.py” file on PyCharm on Windows.
2. Install all dependencies using command

  ```pip install -r requirements.txt```

3. Run on terminal

  ```python clustered_forecast_with_psi.py```

4. Simple click Run on PyCharm

