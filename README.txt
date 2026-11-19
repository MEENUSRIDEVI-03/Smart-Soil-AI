SMART SOIL AI — LIGHT DASHBOARD + SENSOR SIMULATOR

This version is a software-only prototype. It does not connect to an ESP-12E or physical sensors.

LOCAL RUN (Windows)
1. Extract this ZIP.
2. Open Command Prompt in this folder.
3. py -m venv .venv
4. .venv\\Scripts\\activate
5. py -m pip install -r requirements.txt
6. py -m streamlit run dashboard.py

STREAMLIT COMMUNITY CLOUD
- Upload dashboard.py, requirements.txt, and soil_dataset.csv to the root of a GitHub repository.
- Create a Streamlit app using the main branch and dashboard.py as the main file.

SIMULATOR
- Use the slider and click Run simulation, or select one of the quick scenarios.
- Demo mapping follows the original project examples: 150 dry, 600 mid-range, 850 high moisture, 950 very wet.
- Values are simulated and clearly labelled as such.
- The Random Forest trend demonstration unlocks after 30 samples.

NOTES
- Soil-health bands and the soil-type estimate preserve the original prototype's rule-based logic.
- The soil-type estimate is illustrative only; moisture change rate alone cannot reliably identify soil texture.
- Streamlit Community Cloud local files are not guaranteed permanent across app restarts/redeployments.
