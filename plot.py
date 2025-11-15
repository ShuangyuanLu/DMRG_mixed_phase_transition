import matplotlib.pyplot as plt
import numpy as np

# Create a plot
plt.figure()

plt.plot([0, 2], [0, 0], 'k-', label="Horizontal Line")  # 'k-' for a black solid line
plt.plot([1, 1], [-1, 1], 'k-', label="Vertical Line")  # 'k-' for a black solid line

x_list = [1.15, 1.5, 3.75, 4.65]
y_list = [0.5, 0.15, -0.6, 0]
for i in range(4):
    plt.text(x_list[i], y_list[i], '(+)', fontsize=12, ha='center', va='center', color='black')  # ha and va for alignment
x_list = [1.15, 0.5, 3.75, 3.15]
y_list = [-0.5, -0.15, 0.9, 0]
for i in range(4):
    plt.text(x_list[i], y_list[i], '(\u2013)', fontsize=12, ha='center', va='center', color='black')  # ha and va for alignment

x = [1.5 * x + 3 for x in [0, 1, 1, 0, 0]] # Closing the square by repeating the first point
y = [(y - 0.5) * 1.5 for y in [0, 0, 1, 1, 0]]
plt.plot(x, y, 'k-', label='Square')

xy_list = [(1.5, 0), (0.5, 0), (1, 0.5), (1, -0.5), (3.75, -0.75), (4.5, 0), (3.75, 0.75), (3, 0)]
xy_text_list = [(1, 0), (1, 0), (1, 0), (1, 0), (3.5, -0.75), (4.5, -0.5), (4, 0.75), (3, 0.5)]
for i in range(8):
    plt.annotate(
        '',  # No text
        xy=xy_list[i],  # End of the arrow (point)
        xytext=xy_text_list[i],  # Start of the arrow (tail)
        arrowprops=dict(
            arrowstyle="->", lw=1, color='black'  # Arrow style and color
        )
    )


plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.xlim(-0, 5)
plt.ylim(-1, 1)


# Save the plot as a PDF and EPS
plt.savefig("example_plot.pdf", bbox_inches='tight', pad_inches=0.2)  # Save as PDF

plt.show()