import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ground truth data (in meters)
ground_truth = np.array([
    [0.010, 0.020, 0.0, 0.010, 0.010, 0.010],  # Small square: x, y, z, width, height, depth
    [0.070, 0.070, 0.0, 0.020, 0.020, 0.010],  # Medium square
    [0.010, 0.050, 0.0, 0.030, 0.030, 0.010],   # Large square
    [0.050, 0.010, 0.0, 0.040, 0.040, 0.010]   # Large square
])
# Save ground truth to CSV
np.savetxt('ground_truth_squares.csv', ground_truth, 
           header='x(m),y(m),z(m),width(m),height(m),depth(m)', delimiter=',', comments='')
print("✅ Ground truth saved as 'ground_truth_squares.csv'")

# 2D Visualization (top-down view)
plt.figure(figsize=(8, 8))
# Plate surface as dark red
plt.gca().add_patch(plt.Rectangle((0, 0), 0.1, 0.1, linewidth=0, facecolor='#8B0000', alpha=0.7, zorder=0))
# Recesses as white with black outline, drawn on top
for square in ground_truth:
    x, y, _, width, height, _ = square
    rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='white', alpha=1.0, zorder=1)
    plt.gca().add_patch(rect)
plt.xlim(0, 0.1)  # 100mm plate
plt.ylim(0, 0.1)
plt.gca().set_aspect('equal')
plt.title('2D Top-Down View of Plate with Recesses')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True, linestyle='--', alpha=0.3)  # Re-add grid with dashed lines and low opacity

# 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121, projection='3d')
# Plate base (100mm x 100mm x 3mm) as dark red
xx, yy = np.meshgrid(np.linspace(0, 0.1, 10), np.linspace(0, 0.1, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, color='#8B0000', alpha=0.7)
# Recesses as light gray with higher z-order
for square in ground_truth:
    x, y, _, width, height, depth = square
    xx_rec = np.linspace(x, x + width, 5)
    yy_rec = np.linspace(y, y + height, 5)
    XX_rec, YY_rec = np.meshgrid(xx_rec, yy_rec)
    ZZ_rec = np.full_like(XX_rec, -depth)  # Recess depth
    ax.plot_surface(XX_rec, YY_rec, ZZ_rec, color='#D3D3D3', alpha=1.0)  # Light gray
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D View of Plate with Recesses')
ax.set_box_aspect([1, 1, 0.03])  # Adjust z-scale for 3mm thickness
ax.grid(False)  # Keep 3D grid off

plt.tight_layout()
plt.show()

print("\n🎉 SUCCESS! Files ready:")
print("   - ground_truth_squares.csv → Use for labeling tomorrow")
print("   - Visualize to confirm design before printing")