# Offline MMT light-curve files for ACTUAL mode when the live API is unavailable.

Place per-object CSV files here:

```
timestamp,magnitude,error
2024-01-01T00:00:00Z,10.2,0.05
...
```

Or a combined file `mmt_lightcurves.csv` with columns:
`cospar_id`, `object_id`, `timestamp`, `magnitude`, `error`

Files are matched by COSPAR ID or NORAD/object_id substring in the filename.
