#!/usr/bin/env python3
"""
PandaDock Logo Generator

Creates the official PandaDock logo for documentation and branding.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

def create_pandadock_logo():
    """Create the PandaDock logo"""

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Color scheme
    primary_blue = '#1E3A8A'    # Deep blue
    secondary_blue = '#3B82F6'  # Bright blue
    accent_green = '#10B981'    # Emerald green
    gradient_purple = '#8B5CF6' # Purple
    text_color = '#1F2937'      # Dark gray

    # Background gradient effect
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 12, 0, 4], aspect='auto', cmap='Blues', alpha=0.1)

    # Molecular structure representation (stylized)
    # Main molecule backbone
    molecule_x = np.array([1.0, 1.5, 2.2, 2.8, 3.2, 3.8])
    molecule_y = np.array([2.0, 2.4, 2.1, 2.6, 2.2, 2.4])

    # Draw molecular bonds
    for i in range(len(molecule_x)-1):
        ax.plot([molecule_x[i], molecule_x[i+1]], [molecule_y[i], molecule_y[i+1]],
                color=secondary_blue, linewidth=3, alpha=0.8)

    # Draw atoms as circles
    for i, (x, y) in enumerate(zip(molecule_x, molecule_y)):
        if i % 2 == 0:
            color = primary_blue
        else:
            color = accent_green
        circle = Circle((x, y), 0.08, color=color, alpha=0.9)
        ax.add_patch(circle)

    # Protein structure representation (stylized alpha helix)
    helix_x = np.linspace(1.2, 3.5, 100)
    helix_y = 1.0 + 0.3 * np.sin(4 * np.pi * (helix_x - 1.2) / 2.3)
    ax.plot(helix_x, helix_y, color=gradient_purple, linewidth=4, alpha=0.7)

    # Add some side chains to helix
    for i in range(0, len(helix_x), 15):
        x, y = helix_x[i], helix_y[i]
        # Side chain
        ax.plot([x, x + 0.15], [y, y + 0.2], color=gradient_purple, linewidth=2, alpha=0.6)
        ax.plot([x, x - 0.15], [y, y - 0.2], color=gradient_purple, linewidth=2, alpha=0.6)

    # Docking interaction arrows
    arrow_props = dict(arrowstyle='->', lw=2, color=accent_green, alpha=0.8)
    ax.annotate('', xy=(2.0, 1.8), xytext=(2.0, 1.3), arrowprops=arrow_props)
    ax.annotate('', xy=(2.8, 1.6), xytext=(2.8, 1.1), arrowprops=arrow_props)

    # Main title "PANDADOCK"
    ax.text(6.0, 2.8, 'PANDA', fontsize=32, fontweight='bold',
            color=primary_blue, ha='center', va='center',
            family='sans-serif')
    ax.text(8.5, 2.8, 'DOCK', fontsize=32, fontweight='bold',
            color=accent_green, ha='center', va='center',
            family='sans-serif')

    # Subtitle
    ax.text(7.25, 2.1, 'Advanced Molecular Docking Platform',
            fontsize=14, color=text_color, ha='center', va='center',
            style='italic', family='sans-serif')

    # Version and features
    ax.text(7.25, 1.6, 'GPU-Accelerated • High-Throughput • Precision Scoring',
            fontsize=10, color=secondary_blue, ha='center', va='center',
            family='sans-serif')

    # Add decorative elements
    # Grid pattern in background
    for i in range(13):
        ax.axvline(x=i, color='lightgray', alpha=0.2, linewidth=0.5)
    for i in range(5):
        ax.axhline(y=i, color='lightgray', alpha=0.2, linewidth=0.5)

    # Scientific notation decoration
    ax.text(10.5, 3.2, 'E = -RT ln(Ka)', fontsize=10, color=text_color,
            ha='center', va='center', style='italic', alpha=0.6)
    ax.text(10.5, 0.8, 'ΔG = -RT ln(Kd)', fontsize=10, color=text_color,
            ha='center', va='center', style='italic', alpha=0.6)

    # Add some molecular orbital representations
    theta = np.linspace(0, 2*np.pi, 100)
    orbital1_x = 10.8 + 0.2 * np.cos(theta)
    orbital1_y = 2.0 + 0.15 * np.sin(theta)
    ax.plot(orbital1_x, orbital1_y, color=secondary_blue, alpha=0.4, linewidth=1.5)

    orbital2_x = 11.2 + 0.15 * np.cos(theta)
    orbital2_y = 2.0 + 0.2 * np.sin(theta)
    ax.plot(orbital2_x, orbital2_y, color=accent_green, alpha=0.4, linewidth=1.5)

    plt.tight_layout()
    return fig

def create_favicon():
    """Create a simplified favicon version"""
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # Colors
    primary_blue = '#1E3A8A'
    accent_green = '#10B981'

    # Simple molecular representation
    # Central atom
    center = Circle((1.0, 1.0), 0.15, color=primary_blue)
    ax.add_patch(center)

    # Surrounding atoms
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for angle in angles:
        x = 1.0 + 0.4 * np.cos(angle)
        y = 1.0 + 0.4 * np.sin(angle)

        # Bond
        ax.plot([1.0, x], [1.0, y], color=primary_blue, linewidth=3)

        # Atom
        atom = Circle((x, y), 0.08, color=accent_green)
        ax.add_patch(atom)

    # Add 'P' and 'D' letters
    ax.text(1.0, 1.6, 'P', fontsize=20, fontweight='bold',
            color=primary_blue, ha='center', va='center')
    ax.text(1.0, 0.4, 'D', fontsize=20, fontweight='bold',
            color=accent_green, ha='center', va='center')

    plt.tight_layout()
    return fig

def main():
    """Generate all logo variations"""

    # Create main logo
    print("Creating PandaDock main logo...")
    logo_fig = create_pandadock_logo()
    logo_fig.savefig('/Users/pritam/Desktop/PandaDock_Ultimate_Version/docs/pandadock_logo.png',
                     dpi=300, bbox_inches='tight', transparent=True)
    logo_fig.savefig('/Users/pritam/Desktop/PandaDock_Ultimate_Version/docs/pandadock_logo.svg',
                     bbox_inches='tight', transparent=True)
    plt.close(logo_fig)

    # Create favicon
    print("Creating PandaDock favicon...")
    favicon_fig = create_favicon()
    favicon_fig.savefig('/Users/pritam/Desktop/PandaDock_Ultimate_Version/docs/favicon.png',
                        dpi=300, bbox_inches='tight', transparent=True)
    favicon_fig.savefig('/Users/pritam/Desktop/PandaDock_Ultimate_Version/docs/favicon.svg',
                        bbox_inches='tight', transparent=True)
    plt.close(favicon_fig)

    # Create banner version (wide)
    print("Creating PandaDock banner...")
    banner_fig, ax = plt.subplots(1, 1, figsize=(16, 3))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Simple banner with text
    ax.text(8.0, 1.8, 'PANDADOCK', fontsize=48, fontweight='bold',
            color='#1E3A8A', ha='center', va='center')
    ax.text(8.0, 1.0, 'Advanced Molecular Docking Platform',
            fontsize=18, color='#1F2937', ha='center', va='center', style='italic')
    ax.text(8.0, 0.4, 'GPU-Accelerated • High-Throughput • Precision Scoring',
            fontsize=12, color='#3B82F6', ha='center', va='center')

    banner_fig.savefig('/Users/pritam/Desktop/PandaDock_Ultimate_Version/docs/pandadock_banner.png',
                       dpi=300, bbox_inches='tight', transparent=True)
    plt.close(banner_fig)

    print("Logo generation complete!")
    print("Generated files:")
    print("- pandadock_logo.png (main logo)")
    print("- pandadock_logo.svg (vector version)")
    print("- favicon.png (favicon)")
    print("- favicon.svg (vector favicon)")
    print("- pandadock_banner.png (banner version)")

if __name__ == "__main__":
    main()