# Tecator spectroscopy benchmark

Source: StatLib, Carnegie Mellon University.
https://lib.stat.cmu.edu/datasets/tecator

Save the StatLib text file as `Code/data/tecator.txt`. The workspace copy
has an abbreviated header that retains the permission notice and contains
all numerical records. The analysis uses the 100 raw absorbance channels
and fat percentage. StatLib designates rows 1–172 for development and
173–215 for interpolation testing. Seven test rows have the same 100
absorbances as a development row. The primary analysis keeps only the
first occurrence of each spectrum in the original order, leaving 158
development and 35 test spectra. All exact spectral duplicates have
identical fat responses. The original split is reproduced as a benchmark.
The final 25 extrapolation samples are outside this experiment.

The spectra were recorded with a Tecator Infratec Food and Feed Analyzer
(Tecator AB). StatLib includes Tecator's public-domain permission notice
and its request to identify the company and instrument and send a preprint.
