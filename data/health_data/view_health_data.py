import pandas as pd
import plotly.express as px
from datetime import datetime
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Plot heart rate, SpO₂, PI, and Movement over time from a CSV file without headers and save the plots to files.')
parser.add_argument('csv_file', help='Path to the CSV file containing the data.')
parser.add_argument('output_file', help='Base filename for the saved plots (e.g., plot_base_name).')
args = parser.parse_args()

# Define the column names in the correct order
column_names = ['year', 'day_of_year', 'day_of_week', 'hour', 'minute', 'second', 'bpm', 'movement', 'pi', 'spo2']

# Read the data from the CSV file without headers
df = pd.read_csv(args.csv_file, header=None, names=column_names)

# Convert two-digit year to four-digit year
df['year'] = df['year'].apply(lambda x: x + 2000)

# Combine date and time components into a single datetime column
df['datetime'] = df.apply(lambda row: datetime.strptime(
    f"{int(row['year'])}-{int(row['day_of_year'])} {int(row['hour'])}:{int(row['minute'])}:{int(row['second'])}",
    "%Y-%j %H:%M:%S"), axis=1)

# Sort the DataFrame by datetime
df.sort_values('datetime', inplace=True)

# Function to generate and save a plot
def save_plot(df, y_column, title, color, output_filename):
    fig = px.line(
        df,
        x='datetime',
        y=y_column,
        labels={
            'datetime': 'Time',
            y_column: 'Measurement'
        },
        title=title,
        color_discrete_map={y_column: color}
    )
    
    # Customize the layout for a professional look
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Measurement',
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(
            family='Arial, sans-serif',
            size=14,
            color='black'
        ),
        legend=dict(
            title='Parameters',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.05
        ),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        )
    )

    # Add markers to the lines
    fig.update_traces(mode='lines+markers')

    # Save the figure to a file
    fig.write_image(output_filename)
    print(f"Plot saved as image file: {output_filename}")

# Save BPM and SpO₂ plot
save_plot(df, 'bpm', 'Heart Rate (BPM) Over Time', 'red', f"{args.output_file}_bpm.png")
save_plot(df, 'spo2', 'SpO₂ Over Time', 'blue', f"{args.output_file}_spo2.png")

# Save PI and Movement plots
save_plot(df, 'pi', 'Perfusion Index (PI) Over Time', 'green', f"{args.output_file}_pi.png")
save_plot(df, 'movement', 'Movement Over Time', 'purple', f"{args.output_file}_movement.png")

