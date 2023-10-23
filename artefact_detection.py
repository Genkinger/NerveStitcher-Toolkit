import tomllib
import nervestitcher
import numpy
import pandas
import seaborn as sns
from matplotlib import pyplot as plt


artefact_indices = {}
with open("data/artefacts.toml", "rb") as toml:
    artefact_indices = tomllib.load(toml)
