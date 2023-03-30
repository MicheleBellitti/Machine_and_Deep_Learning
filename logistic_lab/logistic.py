import numpy as np
import matplotlib as mpl
from google_drive_downloader import GoogleDriveDownloader
from logistic_regression import LogisticRegression as LR
from data_prepare import get_data, get_bc

X_tr, X_te, Y_tr, Y_te = get_bc()
model = LR(X_tr.shape[1])

model.fit(X_tr, Y_tr, epochs=10000)
model.test(X_te,Y_te)
