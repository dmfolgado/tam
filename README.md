# Time Alignment Measurement for Time Series

Time Alignment Measurement (TAM) is a novel time series distance able to deliver information in the temporal domain, by measuring the ammount of temporal distortion between time series.

#### This repository provides two main contribuctions:
- **Code**: Python implementation of TAM.
- **Dataset**: Contains inertial time series data from Human motion on repetitive tasks. The subjects were asked to perform several repetition of a well-defined task according to different levels of speed.

## How to use TAM?
The value of TAM reflects the amount of time warping between time series. The domain of TAM ranges between 0 (both series are in phase during their complete length) and 1 (both series are completely out-of-phase).

```
import tam
import dtw

d, c, ac, p = dtw.dtw(x, y)
dist = tam.tam(p)
print("Distance %f", dist)
```
