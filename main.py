"""
QuantumBoost ML - Main Entry Point

A Flask web app that forecasts time-series data using a custom
Quantum-Inspired Gradient Optimization algorithm.

================================================================================
README
================================================================================

How to Run:
-----------
    python app.py
    
    Or with Gunicorn:
    gunicorn --bind 0.0.0.0:5000 main:app

Where to Upload Files:
----------------------
    Navigate to http://localhost:5000 and use the upload form on the index page.
    Supported formats: CSV, XLS, XLSX
    
    If no file is uploaded, click "Use Sample Dataset" to generate a demo dataset.

How to Download PDF:
--------------------
    After training completes, click the "Download PDF" button on the results page.
    The PDF includes all metrics, comparison tables, and visualizations.

QuantumBoost Optimizer Parameters:
----------------------------------
    initial_lr (default: 0.01)
        Starting learning rate for gradient updates.
        
    tunneling_rate (default: 0.05)
        Probability of quantum tunneling events that help escape local minima.
        Higher values = more exploration, potentially unstable.
        Recommended range: 0.01 - 0.15
        
    jump_prob (default: 0.03)
        Probability of stochastic jumps for broader exploration.
        Higher values = more random exploration.
        Recommended range: 0.01 - 0.10
        
    temperature (default: 1.0)
        Controls the magnitude of probabilistic jumps.
        Higher values = larger jumps.
        Recommended range: 0.5 - 2.0
        
    decay (default: 0.99)
        Learning rate decay factor per epoch.
        Values closer to 1.0 = slower decay.
        Recommended range: 0.95 - 0.999

================================================================================
"""

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
