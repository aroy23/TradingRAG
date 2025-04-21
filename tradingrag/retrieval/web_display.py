from flask import Flask, render_template
import json
import os
from datetime import datetime

app = Flask(__name__)

def load_results(results_file="rag_results.json"):
    """Load and process the results file."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Separate analysis from results
        analysis = None
        if isinstance(data[-1], dict) and "analysis" in data[-1]:
            analysis = data[-1]["analysis"]
            data = data[:-1]
        
        # Process each result
        for result in data:
            # Format dates
            start_date = datetime.strptime(result['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(result['end_date'], '%Y-%m-%d')
            result['formatted_period'] = f"{start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"
            
            # Add color coding for returns
            result['return_color'] = 'text-success' if result['return'] > 0 else 'text-danger'
            
            # Format similarity score as percentage
            result['similarity_percent'] = f"{result['similarity_score'] * 100:.1f}%"
        
        return data, analysis
    except Exception as e:
        return [], None

@app.route('/')
def display_results():
    """Display the results page."""
    results, analysis = load_results()
    return render_template('results.html', 
                         results=results, 
                         analysis=analysis,
                         timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('tradingrag/retrieval/templates', exist_ok=True)
    app.run(debug=True, port=5000) 