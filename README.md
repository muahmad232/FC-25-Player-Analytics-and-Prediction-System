# FC 25 Player Analytics & Prediction System

This project is a comprehensive Machine Learning suite designed to analyze, predict, and categorize football players based on the EA Sports FC 25 database. It includes predictive models for player positions, overall ratings (OVR), league tiers, and similarity recommender systems for both outfield players and goalkeepers.

## 📂 Project Structure & Models

The system is built across several specialized Jupyter notebooks, each responsible for training specific models or generating feature components.

### 1. Player Position Predictor (`position_predictor.ipynb`)
* **Purpose:** Classifies an outfield player's best position (e.g., ST, CB, CM) based on their raw attributes. It also includes logic to predict a full player profile including position, OVR, and league tier.
* **Model Used:** XGBoost Classifier (`XGBClassifier`) with Grid Search optimization.
* **Key Features:** Custom attribute weighting for different roles (Striker, Midfielder, etc.) to engineer feature scores.
* **Output Files:** `best_xgb_model.pkl`, `position_label_encoder.pkl`, `position_feature_scaler.pkl`.

### 2. Player Overall (OVR) Predictor (`player_overall_predictor.ipynb`)
* **Purpose:** Predicts the Overall Rating (0-99) of an outfield player based on their stats (Pace, Shooting, Passing, etc.).
* **Model Used:** XGBoost Regressor (`XGBRegressor`).
* **Performance:** Achieved an R² score of ~0.985, indicating extremely high accuracy.
* **Output Files:** `best_xgb_regr_oerall.pkl`, `ovr_feature_scaler.pkl`.

### 3. League Tier Predictor (`League_predictor.ipynb`)
* **Purpose:** Classifies which tier of league a player belongs to (Top Tier, Mid Tier, Lower Tier) based on their quality.
* **Model Used:** Random Forest Classifier.
* **Logic:**
    * **Top Tier:** Premier League, LaLiga, Bundesliga, etc.
    * **Mid Tier:** Eredivisie, MLS, Liga Portugal, etc.
    * **Lower Tier:** All others.
* **Output Files:** `league_tier_model.pkl`, `league_tier_encoder.pkl`.

### 4. Outfield Player Similarity (`Similar players.ipynb`)
* **Purpose:** Finds the top 5 most similar players in the world to a given input player using Euclidean distance.
* **Method:** Standard Scaling of 35 distinct player attributes followed by Nearest Neighbor calculation.
* **Output Files:** `similarity_scaler.pkl`, `scaled_attributes.npy`, `player_info.csv`.

### 5. Goalkeeper Analytics (`goal_keepers overall.ipynb` & `goal_keepers_similarity.ipynb`)
* **Purpose:** A dedicated pipeline for Goalkeepers (GK) to predict their OVR and find similar GKs, as they have vastly different stats (Diving, Reflexes, etc.) compared to outfield players.
* **Models Used:** XGBoost Regressor for OVR prediction; Standard Scaler + Euclidean Distance for similarity.
* **Output Files:** `gk_ovr_model.pkl`, `gk_similarity_scaler.pkl`, `scaled_gk_attributes.npy`, `gk_player_info.csv`.

---

## 📊 Dataset

This project uses the **EA Sports FC 25 Database** from Kaggle.
* **Download Link:** [EA Sports FC 25 Database Ratings and Stats](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats)
* **File Used:** `all_players.csv`
* Ensure this file is placed in the root directory before running the notebooks.

---

## 🛠️ Dependencies & Installation

To run these notebooks, you will need Python installed along with the following libraries.

### Core Dependencies
* **Python 3.8+**
* **Pandas:** For data manipulation and CSV reading.
* **NumPy:** For numerical operations and array handling.
* **Scikit-Learn:** For Random Forest models, Scaling, Encoding, and Metrics.
* **XGBoost:** For high-performance Gradient Boosting models.
* **Joblib:** For saving and loading trained models.

### Installation Command
You can install all required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost joblib
