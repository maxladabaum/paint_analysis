# DNA PAINT Picasso-Style ROI Analyzer

This GUI is now map-first:

1. Load a Picasso `*_locs.hdf5` file.
2. Apply drift correction: `none`, `rcc`, or `aim`.
3. Render the stored corrected localization table using Picasso's `picasso.render.render`.
4. Drag a rectangle on the rendered map to select an ROI.
5. Generate histograms from only the corrected localizations inside that ROI.

All histogram analysis is performed after the selected drift correction. If correction method is `none`, the loaded coordinates are treated as the corrected table.

## Start

Double-click:

```text
run_paint_gui.bat
```

The launcher creates a local `.venv` and installs dependencies, including `picassosr`.

## Render And Drift

The map uses Picasso Render directly:

```python
picasso.render.render(...)
```

Correction methods:

- `none`: use loaded coordinates.
- `rcc`: call `picasso.postprocess.undrift`.
- `aim`: call `picasso.aim.aim`.

`Segmentation` is frames per drift segment. AIM also uses `AIM intersect (nm)` and `AIM ROI (nm)`.

Drift correction and rendering are separate actions. Changing `Render pixel`, `Render blur`, or `Min blur` and pressing `Render Map` re-renders the current corrected localization table without rerunning RCC/AIM.

`Min density` and `Max density` match Picasso Render's Display Settings contrast controls: they define which localization density per rendered super-resolution pixel maps to the low and high ends of the colormap. Leave `Max density` at `0` for automatic scaling, equivalent to Picasso's autoscale behavior.

## ROI Histograms

After selecting an ROI on the corrected map, the following plots are generated only from corrected localizations in the ROI:

- photons per localization
- localization precision
- frame number
- fitted PSF sigma x/y
- fitted local background
- nearest-neighbor distance
- linked binding-event length in frames or ms
- linked localizations per event
- linked photons per event

Use `Show Corrected Map` to return to the map after viewing a histogram.

## Linking

Linking has its own settings panel. `Exposure`, `Link radius`, and `Max gap` control event linking after drift correction and ROI selection.

- `Run Linking Analysis` plots a 2x2 summary of event length, event length in ms, localizations per event, and photons per event.
- `Color By Links` draws corrected ROI localizations on the map tab, with multi-localization linked events colored by event ID and single localizations shown in gray.
