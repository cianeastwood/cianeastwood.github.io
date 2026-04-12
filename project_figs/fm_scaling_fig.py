import matplotlib.pyplot as plt

# -----------------------------
# 1. DATA PREPARATION
# -----------------------------

# Prior Works
# Format: (Name, ExaFLOPs, FID)
# Note: Values for IMPA and PhenDiff adjusted to match the visual log scale positions in the image
prior_works = [
    ("IMPA", 0.5, 41.6),
    ("CellFlux", 7.83, 33.0),
    ("PhenDiff", 0.8, 68.0),
    ("CellFluxV2", 7.9, 19.0),
]

# "Ours"
ours_data = [
    ("$\\mathcal{N} \\rightarrow \\mathcal{D}$ (undertrained)", 3.93, 12.3),
    ("$\\mathcal{N} \\rightarrow \\mathcal{D}$", 15.7, 9.90),
    ("MiT-XL/2", 118.0, 9.07),
    ("MiT-XL/2\n+ Pretraining", 1000.0, 8.8),
]

# Separate 'Ours' into lists for plotting
ours_x = [item[1] for item in ours_data]
ours_y = [item[2] for item in ours_data]
ours_labels = [item[0] for item in ours_data]

# Separate 'Prior Works' into lists
prior_x = [item[1] for item in prior_works]
prior_y = [item[2] for item in prior_works]
prior_labels = [item[0] for item in prior_works]

# -----------------------------
# 2. PLOTTING
# -----------------------------

# Figure setup
plt.figure(figsize=(4.5, 4.25))
plt.rcParams['font.family'] = 'sans-serif'

# Colors pulled from the screenshot
color_prior = 'tab:blue'      #'#1b30b0'
color_ours = 'tab:orange'      #'#aa5ce3'

# Plot Prior Works (Scatter only)
plt.scatter(prior_x, prior_y, color=color_prior, s=160, label='Prior Work', zorder=5)

# Plot "Ours" (Line + Scatter)
plt.plot(ours_x, ours_y, color=color_ours, linewidth=2.5, zorder=4)
plt.scatter(ours_x, ours_y, color=color_ours, s=160, label='Ours', zorder=6)

# -----------------------------
# 3. ANNOTATIONS
# -----------------------------

# Annotate Prior Works
for name, x, y in zip(prior_labels, prior_x, prior_y):
    if name == "CellFlux":
        xytext = (0, 10)
        va = 'bottom'
        ha = 'center'
    elif name == "CellFluxV2":
        xytext = (0, -15)
        va = 'top'
        ha = 'center'
    else: # IMPA, PhenDiff
        xytext = (10, 0)
        va = 'center'
        ha = 'left'
        
    plt.annotate(
        name, 
        (x, y), 
        xytext=xytext, 
        textcoords='offset points', 
        fontsize=11, 
        color=color_prior, 
        fontweight='bold',
        va=va, ha=ha
    )

# # Annotate "Ours"
# for name, x, y in zip(ours_labels, ours_x, ours_y):
#     if "undertrained" in name:
#         xytext = (0, 12)
#         va = 'bottom'
#         ha = 'center'
#     elif name == "$\\mathcal{N} \\rightarrow \\mathcal{D}$":
#         xytext = (5, 8)
#         va = 'bottom'
#         ha = 'left'
#     elif "MiT-XL/2\n+" in name:
#         xytext = (0, 12)
#         va = 'bottom'
#         ha = 'center'
#     elif "MiT-XL/2" in name:
#         xytext = (-8, 8)
#         va = 'bottom'
#         ha = 'right'

#     plt.annotate(
#         name, 
#         (x, y), 
#         xytext=xytext, 
#         textcoords='offset points', 
#         fontsize=11, 
#         color=color_ours, 
#         fontweight='bold',
#         va=va, ha=ha
#     )

# -----------------------------
# 4. FORMATTING
# -----------------------------

# Set Log Scale
plt.xscale('log')
plt.yscale('log')

# Set Grid (styled to match the image's faint lines)
plt.grid(True, which="major", ls="-", color='#D9D9D9', alpha=0.8)
plt.grid(True, which="minor", ls=":", color='#E5E5E5', alpha=0.6)

# Axis Labels
plt.xlabel("Total Training Compute (ExaFLOPs)", fontsize=14)
plt.ylabel("FID (Lower is better)", fontsize=14)

# Explicitly set X and Y ticks to match the exact gaps seen in the screenshot
plt.xticks([1, 5, 100, 1000], ["1", "5", "100", "1000"], fontsize=12)
plt.yticks([10, 20, 30, 40, 50, 70], ["10", "20", "30", "40", "50", "70"], fontsize=12)

# Remove top and right box boundaries for a cleaner aesthetic
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Clean Legend
plt.legend(loc='upper right', frameon=False, fontsize=14, handletextpad=0.1)

# Set the limits of the plot to properly fit all annotations
plt.xlim(0.3, 1600)
plt.ylim(7, 74)

# Save
plt.tight_layout()
plt.savefig("fm_2.svg", bbox_inches="tight")
# plt.show()