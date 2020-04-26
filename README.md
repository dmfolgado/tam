# Time Alignment Measurement for Time Series

Time Alignment Measurement (TAM) is a novel time series distance able to deliver information in the temporal domain, by measuring the ammount of temporal distortion between time series.


In order to use our proposed measurement please cite the following: Folgado, Duarte, et al. "Time Alignment Measurement for Time Series." Pattern Recognition 81 (2018): 268-279.

#### This repository provides two main contribuctions:
- **Code**: Python implementation of TAM.
- **Dataset**: Contains inertial time series data from Human motion on repetitive tasks. The subjects were asked to perform several repetition of a well-defined task according to different levels of speed.

## Data presentation
The dataset contains inertial information retrieved by six different subjects that executed ten repetitions of a well-defined task under four distinct sets. Difference among sets resides in the speed the subject is accomplishing the task.
The movements performed during each task consisted of: grasping a solderless breadboard used to build electronic circuits; placing the board on a defined position and welding a single perforation in each repetition; grasping the welded board and move it to a defined position.

An example of accelerometer, gyroscope and magnetometer (magnitude) data is presented on the following figure:

![Kiku](imgs/data_presentation.png)

## How to use TAM?
The value of TAM reflects the amount of time warping between time series. The domain of TAM ranges between 0 (both series are in phase during their complete length) and 3 (both series are completely out-of-phase).

```
import tam
import dtw

d, c, ac, p = dtw.dtw(x, y)
dist = tam.tam(p)
print("Distance %f", dist)
```
