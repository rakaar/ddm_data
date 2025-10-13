# CRSY25 Figure Export Task Documentation

## Task Overview
Export individual subfigures from multi-panel matplotlib figures into separate files in multiple publication-ready formats.

## Required Formats
Each subfigure must be saved in **4 formats**:
1. **PDF** - Vector format for publications
2. **EPS** - Vector format for some journals/publishers
3. **SVG** - Vector format for web/editing
4. **PNG** - High-resolution raster (600 DPI) for presentations/previews

## User Instructions
When requesting this task, provide:
1. **Path to the figure script** (e.g., `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/make_fig1_CRSY25.py`)
2. **Which subfigures to save** - Be specific about which panels/combinations you want
   - Example: "4 psychometric panels", "2 JND plots", "all chronometric + summary plots"
3. **Output directory preference** (optional - default: `crsy_25_figs_new/fig1/`)

## Common Issues & Solutions

### Issue 1: Labels Cut Off (Most Common)
**Symptoms**: Y-axis labels, titles, or figure-level text missing from saved figures

**Solutions**:
1. **Increase padding** in the save function:
   ```python
   save_multiple_subfigures(axes, "filename", expand_left=1.5, expand_right=1.35)
   ```

2. **Include figure-level text artists**:
   ```python
   ylabel_text = fig.text(x, y, 'Label Text', ...)
   save_multiple_subfigures(axes, "filename", extra_artists=[ylabel_text])
   ```

3. **Common padding values**:
   - Default: `expand_left=1.25, expand_right=1.25`
   - With y-labels: `expand_left=1.5, expand_right=1.25`
   - With both sides: `expand_left=1.5, expand_right=1.35`

### Issue 2: Permission Denied
**Symptoms**: `PermissionError: [Errno 13] Permission denied`

**Solution**: Change output directory ownership or use a different directory
```bash
# Check current ownership
ls -la output_directory/

# Solution 1: Use a new directory with correct permissions
OUTPUT_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/new_output/'

# Solution 2: Fix permissions (if you have sudo access)
sudo chown -R $USER:$USER output_directory/
```

### Issue 3: Figure Elements Not Aligned
**Symptoms**: Axes don't line up properly, inconsistent spacing

**Solution**: Always call `fig.canvas.draw()` before saving
```python
# After all positioning/alignment adjustments
fig.canvas.draw()
# Then save
save_multiple_subfigures(...)
```

### Issue 4: Empty or Incomplete Figures
**Symptoms**: Saved figure is blank or missing elements

**Causes & Solutions**:
1. **Data not loaded**: Ensure pickle/CSV files exist before saving
2. **Axes not rendered**: Call `fig.canvas.draw()` before saving
3. **Wrong axes list**: Verify you're passing the correct axes to the function

## Bounding Box Calculation Details

The function automatically:
1. Gets window extent of each axis in inches
2. Combines all bounding boxes into one
3. Expands asymmetrically:
   - Left/right: `(width * (expand_factor - 1)) / 2`
   - Top/bottom: `height * 0.175` (17.5% each side, 35% total)

## Testing Checklist

Before finalizing, verify:
- [ ] All 4 formats generated for each subfigure
- [ ] Labels fully visible (not cut off)
- [ ] File sizes reasonable (PNG ~100KB-2MB, PDF ~20-70KB)
- [ ] PNG resolution is 600 DPI
- [ ] Filenames are descriptive and consistent
- [ ] White background (`facecolor='white'`)


## Environment Setup

Use the virtual environment:
```bash
cd /home/rlab/raghavendra/ddm_data/fit_animal_by_animal
/home/rlab/raghavendra/ddm_data/.venv/bin/python your_script.py
```

## Output Verification

After running, check:
```bash
# List files
ls -lh output_directory/

# Check PNG dimensions and DPI
identify -format "%f: %wx%h, DPI: %x x %y\n" output_directory/*.png

# Count files (should be 4 × number of subfigures)
ls output_directory/ | wc -l
```

## Notes for AI Assistant

When this task is requested:
1. Read the target Python script to understand its structure
2. Identify all subplot axes being created (ax1, ax2, etc.)
3. Ask user which subfigure combinations they want saved
4. Add the `save_multiple_subfigures()` function if not present
5. Add save calls after each subplot section with appropriate padding
6. Test by running the script and checking output
7. Adjust padding if labels are cut off
8. Always use `fig.canvas.draw()` before saving

## Reference Implementation
See: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/make_fig1_CRSY25.py`
- Lines 32-89: Function definition
- Lines 248-251: Psychometric save example
- Lines 637-642: JND save example (with extra padding)
- Lines 505-508: Mean RT save example (with extra artists)
