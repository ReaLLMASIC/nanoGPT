import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI issues in threads

import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import io
import imageio.v2 as imageio
import matplotlib.image as mpimg
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load the background image once and reuse it across all frames
background_image = None

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def parse_time_data(row):
    """Extracts the quarter, game clock, minutes, seconds, and shot clock from a row."""
    quarter = int(row[0][1:])
    game_clock = float(row[1][1:])
    minutes = int(game_clock) % 3600 // 60
    seconds = int(game_clock) % 60
    shot_clock = float(row[4][1:])
    return quarter, game_clock, minutes, seconds, shot_clock

def parse_player_data(row):
    players = {'home': [], 'away': []}
    for item in row[5:]:
        if item.startswith('H'):
            parts = item.split('X')
            number = int(parts[0][1:])
            x = float(parts[1].split('Y')[0])
            y = float(parts[1].split('Y')[1])
            players['home'].append((number, x, y))
        elif item.startswith('A'):
            parts = item.split('X')
            number = int(parts[0][1:])
            x = float(parts[1].split('Y')[0])
            y = float(parts[1].split('Y')[1])
            players['away'].append((number, x, y))
    return players

def parse_ball_data(row):
    bx = float(row[-3][2:])
    by = float(row[-2][2:])
    bz = float(row[-1][2:])
    return bx, by, bz

def initialize_background():
    global background_image
    if background_image is None:
        background_image = mpimg.imread('court.png')

def create_frame(players, ball, player_size, ball_size, ball_scale_with_z, quarter, minutes, seconds, shot_clock):
    # Reuse the global background image
    initialize_background()

    fig, ax = plt.subplots(figsize=(9.39, 5.0))  # Set figure size according to the desired aspect ratio
    ax.set_xlim(0, 100)  # X-axis remains from 0 to 100
    ax.set_ylim(0, 53.24)  # Adjust Y-axis to fit the aspect ratio
    ax.set_aspect(1.878)  # Set aspect ratio to match the 939x500 ratio

    # Display the background image
    ax.imshow(background_image, extent=[0, 100, 0, 53.24], aspect='auto')

    # Plot home players
    for number, x, y in players['home']:
        circle = Circle((x, y), player_size, facecolor='blue', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, str(number), ha='center', va='center', color='white')

    # Plot away players
    for number, x, y in players['away']:
        circle = Circle((x, y), player_size, facecolor='red', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, str(number), ha='center', va='center', color='white')

    # Plot ball
    bx, by, bz = ball
    if ball_scale_with_z:
        ball_radius = ball_size * (bz / 100 + 0.5)  # Scale between 0.5x and 1.5x of ball_size
    else:
        ball_radius = ball_size
    ball_circle = Circle((bx, by), ball_radius, facecolor='orange', edgecolor='black')
    ax.add_patch(ball_circle)

    # Add time information (quarter, game clock, shot clock)
    time_text = f"Basketball Forecast Visualization\nQuarter: {quarter}, Time: {minutes:02d}:{seconds:02d}, Shot Clock: {shot_clock:.1f}s"
    ax.text(0.5, 1.05, time_text, transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')

    # plt.title(f"Basketball Forecast Visualization")

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = imageio.imread(buf)
    buf.close()
    plt.close(fig)

    return image

def process_frame(i, row, player_size, ball_size, ball_scale_with_z):
    # Extract time information
    quarter, game_clock, minutes, seconds, shot_clock = parse_time_data(row)
    players = parse_player_data(row)
    ball = parse_ball_data(row)
    image = create_frame(players, ball, player_size, ball_size, ball_scale_with_z, quarter, minutes, seconds, shot_clock)
    return i, image  # Return index and image

def create_gif(data, output_file, player_size, ball_size, ball_scale_with_z, fps, num_threads=4):
    results = []

    # Use multithreading to speed up frame generation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_frame, i, row, player_size, ball_size, ball_scale_with_z)
            for i, row in enumerate(data)
        ]

        # Process frames and collect images using tqdm progress bar
        for future in tqdm(futures, desc="Processing frames"):
            results.append(future.result())

    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])
    images = [image for i, image in results]
    imageio.mimsave(output_file, images, fps=fps)
    print(f"GIF saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize basketball player and ball positions with time information and create a GIF.")
    parser.add_argument("csv_file", help="Path to the CSV file containing position data.")
    parser.add_argument("--player_size", type=float, default=2.0, help="Size of player circles.")
    parser.add_argument("--ball_size", type=float, default=1.0, help="Base size of the ball circle.")
    parser.add_argument("--ball_scale", action="store_true", help="Scale ball size with z-coordinate.")
    parser.add_argument("--output", default="basketball_animation.gif", help="Output GIF file name.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use.")
    args = parser.parse_args()

    data = read_csv(args.csv_file)
    create_gif(data, args.output, args.player_size, args.ball_size, args.ball_scale, args.fps, args.threads)

if __name__ == "__main__":
    main()

