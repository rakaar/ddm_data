---
description: Make figure panels smaller and square (like fig2 changes in e3d0385)
---

# Steps to resize panels in a multi-panel figure

1. Find the `FigureBuilder` instantiation and change `figsize` to a smaller square value (e.g., `(9, 9)` instead of `(12, 10)`)

2. Increase `hspace` and `wspace` slightly (e.g., from `0.3` to `0.4`) to accommodate labels

3. For each subplot, add `ax.set_box_aspect(1)` immediately after creating the axes to force square panels:
   ```python
   ax = builder.fig.add_subplot(builder.gs[row, col])
   ax.set_box_aspect(1)
   plot_function(ax, data)
   ```

4. Run the script to regenerate the figure and verify panels are smaller and square

5. Provide scp commands to transfer the generated png/pdf files and modified python script to the user's local machine at `/home/rka/Downloads/scp_files/` (use /scp workflow pattern)
