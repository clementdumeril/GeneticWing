# Weather grids of real days

Each `<site>_<date>.npz` holds the archived forecast of one day (08:00 to 19:00 local time) on a 40 km x 30 km grid
with a 2.5 km step, centred on the site, as written by `geneticwing.weather.fetch_day`:

- DWD ICON-D2 (2.2 km): surface sensible heat flux, 2 m temperature and dew point, low cloud cover, precipitation,
  temperature and geopotential height from 975 to 700 hPa, wind at 925 hPa;
- NOAA GFS (0.25 deg): boundary-layer height at the site, to check the parcel method.

| File | Day | Midday weather (14:00) |
| --- | --- | --- |
| `issoudun_2025-07-12.npz` | strong, blue | w* 2.9 m/s, z_i 2,280 m (GFS 2,450), no low cloud, wind 5.6 m/s |
| `issoudun_2025-08-02.npz` | medium, cumulus | w* 2.2 m/s, z_i 1,730 m (GFS 1,460), cloud base 1,550 m, 37 % low cloud, wind 4.3 m/s |
| `issoudun_2025-06-22.npz` | windy, shallow | w* 1.9 m/s, z_i 890 m (GFS 1,525), 12 % low cloud, wind 6.7 m/s |

Wind at 925 hPa; z_i by the parcel method, the GFS boundary-layer height in brackets. The two disagree most on the
windy day, where mechanical mixing deepens GFS's layer beyond the dry thermals' top.

Issoudun-Le Fay (46.89 N, 2.04 E) is a gliding centre in flat country (ground 123 to 203 m in the grid).

Data: Open-Meteo historical forecast API (https://open-meteo.com), licensed CC BY 4.0; ICON-D2 from Deutscher
Wetterdienst, GFS from NOAA.
