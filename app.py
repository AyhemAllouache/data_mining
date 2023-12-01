from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64

# Load and preprocess the dataset
data = pd.read_csv('card-credit.csv')
selected_columns = data[['PURCHASES', 'CASH_ADVANCE']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_columns)

# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_features)
# Initialize Flask app
app = Flask(__name__)


def create_plot(new_point_scaled, scaled_features, kmeans_clusters, feature1_name, feature2_name):
    plt.figure(figsize=(10, 6))

    # Unique cluster labels
    unique_clusters = np.unique(kmeans_clusters)

    # Plot each cluster with a label for the legend
    for cluster in unique_clusters:
        plt.scatter(
            scaled_features[kmeans_clusters == cluster, 0],
            scaled_features[kmeans_clusters == cluster, 1],
            label=f'Cluster {cluster}',
            alpha=0.6
        )

    cluster_label = None
    # Plot new data point if provided
    if new_point_scaled is not None:
        # Predict the cluster for new data point
        cluster_label = kmeans.predict(new_point_scaled)[0]
        plt.scatter(
            new_point_scaled[0, 0],
            new_point_scaled[0, 1],
            c='red',
            marker='x',
            s=100,
            label='New Data Point'
        )

    # Add legend
    plt.legend()

    # Label axes with the real feature names
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)

    # Add a title with a message about the new point's cluster
    if cluster_label is not None:
        plt.title(f'The new Point Belongs to Cluster {cluster_label}')
    else:
        plt.title('Cluster Plot')

    # Convert plot to PNG image
    png_output = io.BytesIO()
    plt.savefig(png_output, format='png', bbox_inches='tight')
    plt.close()
    png_output.seek(0)
    plot_data = base64.b64encode(png_output.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{plot_data}"






@app.route('/')
def welcome():
    # Render the welcome page
    return render_template('Welcome.html')


@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_page():
    plot_url = None
    if request.method == 'POST':
        # Get selected features from the form
        feature1 = request.form['feature1']
        feature2 = request.form['feature2']
        # Get values for the selected features
        value1 = float(request.form['value1'])
        value2 = float(request.form['value2'])
        
        # Load and preprocess the dataset
        data = pd.read_csv('card-credit.csv')
        selected_columns = data[[feature1, feature2]].dropna()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_columns)

        # Train K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_clusters = kmeans.fit_predict(scaled_features)
        
        # Scale the new point
        new_point = np.array([[value1, value2]])
        new_point_scaled = scaler.transform(new_point)

        # Create the plot with the new point
        plot_url = create_plot(new_point_scaled, scaled_features, kmeans_clusters,feature1, feature2)

    # Render the main page with plot
    return render_template('Kmeans.html', plot_url=plot_url)
if __name__ == '__main__':
    app.run(debug=True)
