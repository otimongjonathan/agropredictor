# Hybrid TemporalKG Agricultural Cost Predictor

A Flask-based web application that uses a hybrid AI model combining Temporal (GRU) and Knowledge Graph embeddings to predict agricultural input costs for various crops in Uganda.

## Features

- **Hybrid AI Model**: Combines GRU temporal patterns with Knowledge Graph embeddings
- **Real-time Predictions**: Predict individual input costs (seeds, fertilizer, herbicide, pesticide, labor) per acre
- **Cascading Dropdowns**: Region → District → Crop selection with data validation
- **Multipliers Display**: Shows crop-specific input multipliers per acre
- **Future Predictions**: Predict costs for next 3 months
- **Beautiful UI**: Modern, responsive web interface

## Model Architecture

- **Temporal Component (GRU)**: Learns historical price patterns over 6-month sequences
- **Knowledge Graph Component**: Captures relationships between Regions, Districts, and Crops using embeddings
- **Fusion Layer**: Intelligently combines both components

## Model Performance

- Test R²: 99.89%
- Test RMSE: 365.49 UGX

## Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd myapp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files exist**
   - `models/best_normalized_model.pth` - Trained model weights
   - `models/normalized_preprocessing.pkl` - Preprocessing pipeline
   - `train_dataset_cleaned.csv` - Training data

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open browser: `http://localhost:5000`

### Deployment

See `DEPLOYMENT.md` for detailed deployment instructions to Crane Cloud.

## Project Structure

```
myapp/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── Procfile                    # Gunicorn server config (for deployment)
├── runtime.txt                 # Python version specification
├── templates/
│   └── index.html             # Frontend UI
├── models/
│   ├── best_normalized_model.pth      # Trained model weights
│   └── normalized_preprocessing.pkl   # Preprocessing pipeline
├── train_dataset_cleaned.csv          # Training data (for dropdowns)
├── train_models.py                     # Model training script
└── create_preprocessing.py             # Preprocessing creation script
```

## API Endpoints

### GET `/`
- Returns the main web UI

### POST `/predict`
- **Request Body:**
  ```json
  {
    "region": "Central",
    "district": "Kampala",
    "crop": "Coffee",
    "rainfall": 75,
    "soil_fertility": 80
  }
  ```
- **Response:**
  ```json
  {
    "predicted_cost": 68184.0,
    "individual_prices": {
      "seed_price": 813250.0,
      "fertilizer_price": 849040.0,
      "herbicide_price": 98978.5,
      "pesticide_price": 64723.2,
      "labor_cost": 123824.0,
      "unit_prices": {...},
      "multipliers": {...}
    },
    "future_predictions": [...]
  }
  ```

### GET `/regions`
- Returns list of all available regions

### GET `/districts?region=<region_name>`
- Returns districts filtered by region (optional)

### GET `/crops?district=<district_name>&region=<region_name>`
- Returns crops filtered by district and/or region (optional)

### GET `/multipliers`
- Returns multipliers per acre for all crops

## Technologies Used

- **Backend**: Flask, Python
- **AI/ML**: PyTorch, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn, Crane Cloud

## Dependencies

- torch>=1.9.0,<2.0.0
- flask>=2.0.0
- numpy>=1.21.0,<2.0.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- joblib>=1.0.0
- gunicorn>=20.0.0

## Development

### Training Models

To train new models or retrain existing ones:
```bash
python train_models.py
```

### Creating Preprocessing

To create preprocessing pipeline:
```bash
python create_preprocessing.py
```

## License

[Your License Here]

## Author

[Your Name/Organization]

## Support

For issues or questions, please open an issue in the repository.

