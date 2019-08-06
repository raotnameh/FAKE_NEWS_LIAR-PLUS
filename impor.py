
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
import json
import re
import glob
import os
import string
import random
import requests
import scipy
from matplotlib.colors import *
import seaborn as sn
from dateutil.parser import parse
import datetime as dt
pd.options.mode.chained_assignment = None  # default='warn'

# Sklearn imports
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.utils import shuffle
from sklearn.utils.class_weight import *
from sklearn.svm import *
from sklearn.externals import *
from scipy.stats import *

# For sentiment analysis
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from google.cloud import language

from tqdm import tqdm

# Import for WordCloud
import wordcloud

# Text classifier - TextBlob
from textblob.classifiers import NaiveBayesClassifier

# Local imports
from helpers import *
import cleaningtool as ct