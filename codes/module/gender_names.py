import pandas as pd

# load name from gender computer
gcm = pd.read_csv("../../asset/gender_computer/male_names_only_USA.csv")
gcm = gcm.sample(frac=1, random_state=123)
mnames = gcm["name"].tolist()  # male names from USA

gcf = pd.read_csv("../../asset/gender_computer/female_names_only_USA.csv")
gcf = gcf.sample(frac=1, random_state=123)
fnames = gcf["name"].tolist()  # female names from USA

# small name for debugging
# mnames = ["Alonzo", "Adam"]
# fnames = ["Ebony", "Amanda"]




