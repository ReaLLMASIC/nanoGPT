import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table
import argparse

def view_csv(csv_path, digits):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize the console
    console = Console()
    
    # Create a table
    table = Table(title="Results from CSV")
    table.add_column("A", justify="right", style="cyan")
    table.add_column("Min Loss", justify="right", style="magenta")
    table.add_column("Max Loss", justify="right", style="magenta")
    table.add_column("Mean Loss", justify="right", style="magenta")
    table.add_column("Median Loss", justify="right", style="magenta")
    table.add_column("Std Dev", justify="right", style="magenta")

    # Add rows to the table
    for _, row in df.iterrows():
        table.add_row(
            str(row["A"]),
            f"{row['min']:.{digits}f}",
            f"{row['max']:.{digits}f}",
            f"{row['mean']:.{digits}f}",
            f"{row['median']:.{digits}f}",
            f"{row['std']:.{digits}f}"
        )

    # Print the table
    console.print(table)

def plot_trends(df):
    # Scatter plot for min loss vs. A
    fig = px.scatter(df, x='A', y='min', title='Scatter Plot of Min Loss vs. A')
    fig.show()

    # Line plot for mean loss with standard deviation bounds vs. A
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['A'], y=df['mean'], mode='lines+markers', name='Mean Loss',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['A'], y=df['mean'] + df['std'], mode='lines', name='Mean + Std Dev',
        line=dict(color='lightblue'), fill='tonexty'
    ))
    fig.add_trace(go.Scatter(
        x=df['A'], y=df['mean'] - df['std'], mode='lines', name='Mean - Std Dev',
        line=dict(color='lightblue'), fill='tonexty'
    ))
    fig.update_layout(title='Mean Loss with Standard Deviation Bounds vs. A')
    fig.show()

    # Box plot for each A
    loss_columns = [col for col in df.columns if col.startswith('loss')]
    melted_df = df.melt(id_vars=['A'], value_vars=loss_columns, var_name='loss', value_name='value')
    fig = px.box(melted_df, x='A', y='value', title='Box Plot of Losses for each A')
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="View CSV results with rich formatting and plot trends.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file.")
    parser.add_argument('--digits', type=int, default=4, help="Number of digits to display for loss values.")
    parser.add_argument('--plot', action='store_true', help="Plot trends using Plotly.")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    view_csv(args.csv_path, args.digits)
    
    if args.plot:
        plot_trends(df)

if __name__ == "__main__":
    main()

