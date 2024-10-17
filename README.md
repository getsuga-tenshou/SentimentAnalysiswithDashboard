# Real-Time Conflict and Sentiment Analysis Pipeline

## Overview

This project implements an incremental data processing pipeline designed to analyze global conflicts and sentiments in real-time using the **GDELT Event 2.0** dataset. The system extracts data on international events, applies feature engineering, and utilizes both supervised and unsupervised learning techniques to classify conflicts and assess sentiments. Results are visualized in a user-friendly **Power BI dashboard**.

## Features

- **Real-Time Data Processing**: The pipeline retrieves and processes new data from the GDELT dataset every 15 minutes, ensuring real-time updates on global events.
- **Conflict Classification**: Events are classified as either **verbal** or **material conflicts**, with further distinction between **internal** and **external conflicts**.
- **Sentiment Analysis**: Events are analyzed for sentiment (positive or negative) based on the **Goldstein scale** and **AvgTone**.
- **Dashboard Visualization**: A comprehensive **Power BI dashboard** visualizes global conflicts, showing key metrics such as sentiment, event magnitude, and country involvement.

## Pre-requisite Libraries:

To install the necessary libraries, run:

```bash
pip install google-cloud-bigquery
pip install pandas-gbq
pip install pandas
pip install scikit-learn
```

## Setup Instructions

1. **Google Cloud Setup**

   - Make sure you have a **Google Cloud Project** with **BigQuery** enabled.
   - Create a **Service Account** and download the `service-account.json` file.

2. **Add `service-account.json`**

   - **Do not commit the `service-account.json` file to the repository**.
   - Place the downloaded `service-account.json` file in the root of the project directory.

   Example structure:

   ```
   /project-root
     ├── service-account.json
     ├── analyser.py
   ```

3. **Run the Application**
   - Once the credentials are in place, you can run the project normally:
     ```bash
     python analyser.py
     ```

## Contribution

- Contributions to this project were made by:
  - [@jawad774](https://github.com/jawad774)
  - [@cymtrick](https://github.com/cymtrick)
  - [@getsuga-tenshou](https://github.com/getsuga-tenshou)
