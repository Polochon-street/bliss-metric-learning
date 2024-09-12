# TODO: test that the triplets are what they should be
import sqlite3
import os
from xdg.BaseDirectory import xdg_data_home

# Regularization code for the survey (~600 answers).
# Works best with lambdas between 0 and 0.1.
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import approx_fprime, check_grad, minimize

from scipy.stats import norm
from scipy.spatial.distance import norm as L2_norm


import numpy as np

database_path = "test.db"
con = sqlite3.connect(database_path)
cur = con.cursor()

# Absolutely shameless code
query = cur.execute(
    """
    select song_id, training_triplet.id, 1 as song_number, feature, feature.feature_index from feature
    inner join training_triplet on feature.song_id = training_triplet.song_1_id
    union all
    select song_id, training_triplet.id, 2 as song_number, feature, feature.feature_index from feature
    inner join training_triplet on feature.song_id = training_triplet.song_2_id
    union all
    select song_id, training_triplet.id, 3 as song_number, feature, feature.feature_index from feature
    inner join training_triplet on feature.song_id = training_triplet.odd_one_out_id
    order by training_triplet.id, song_number, feature.feature_index
    """
)
# Copy list so we don't have an iterator
query = [x for x in query]

song2_analysis = np.array(
    [
        0.29539990425109863,
        -0.6704341173171997,
        -0.5210916996002197,
        -0.8818870782852173,
        -0.26809781789779663,
        -0.6857472658157349,
        -0.11365640163421631,
        -0.7554588913917542,
        0.721436619758606,
        0.6880323886871338,
        0.28592443466186523,
        -0.04419243335723877,
        -0.05082428455352783,
        -0.013398408889770508,
        0.24753963947296143,
        -0.07814419269561768,
        -0.9295040965080261,
        -0.9361764192581177,
        -0.9459710121154785,
        -0.942000687122345,
    ]
)
song3_analysis = np.array(
    [
        0.14403247833251953,
        -0.8214370608329773,
        -0.6280502080917358,
        -0.8501287698745728,
        -0.6089711785316467,
        -0.6492020487785339,
        -0.3009887933731079,
        -0.7216899394989014,
        0.6692289113998413,
        0.6182026863098145,
        0.24064040184020996,
        -0.20484429597854614,
        -0.1482003927230835,
        -0.14023298025131226,
        0.12168753147125244,
        -0.16945737600326538,
        -0.9439558982849121,
        -0.9446036219596863,
        -0.9544410109519958,
        -0.952128529548645,
    ]
)
song1_analysis = np.array(
    [
        0.5244531631469727,
        -0.8197064995765686,
        -0.6663101315498352,
        -0.8554021120071411,
        -0.6086487174034119,
        -0.7205168604850769,
        -0.39685624837875366,
        -0.6972397565841675,
        0.8513143062591553,
        0.7946159839630127,
        0.17041051387786865,
        -0.18601077795028687,
        -0.22524899244308472,
        -0.20525473356246948,
        -0.06885206699371338,
        -0.29495465755462646,
        -0.9507571458816528,
        -0.9507496356964111,
        -0.9587293267250061,
        -0.9564813375473022,
    ]
)

song1_1_analysis = [0.06333267688751221,
 -0.7679155468940735,
 -0.5530778765678406,
 -0.7974168658256531,
 -0.42773473262786865,
 -0.5261541604995728,
 -0.20134001970291138,
 -0.6503257751464844,
 0.6831680536270142,
 0.6726655960083008,
 0.5156009197235107,
 -0.06994158029556274,
 -0.11051088571548462,
 -0.05550694465637207,
 0.013588905334472656,
 -0.13684529066085815,
 -0.9398207068443298,
 -0.9434391260147095,
 -0.9489748477935791,
 -0.9433183670043945]
song1_2_analysis = [
    0.5094703435897827,
 -0.7536754608154297,
 -0.5465043783187866,
 -0.8614546060562134,
 -0.36992013454437256,
 -0.6044321060180664,
 -0.13895171880722046,
 -0.7266719937324524,
 0.7491354942321777,
 0.710315465927124,
 0.4252873659133911,
 -0.07529580593109131,
 -0.07063114643096924,
 0.004880785942077637,
 0.17831110954284668,
 -0.023372232913970947,
 -0.930760383605957,
 -0.9344757795333862,
 -0.9425827860832214,
 -0.9374039173126221]
song1_3_analysis = [0.3349463939666748,
 -0.8952140212059021,
 -0.7330068349838257,
 -0.8772907853126526,
 -0.806042492389679,
 -0.8004536628723145,
 -0.4704028367996216,
 -0.7303187847137451,
 0.6941958665847778,
 0.6760915517807007,
 0.26948463916778564,
 -0.15575885772705078,
 -0.19588935375213623,
 -0.22775810956954956,
 -0.187822163105011,
 -0.24095487594604492,
 -0.9586858749389648,
 -0.9598385691642761,
 -0.9593982696533203,
 -0.9625996351242065]



triplets = []

ids = set(t for _, t, _, _, _ in query)

for id in ids:
    current_triplet_list = [(i, p, f) for i, tid, p, f, _ in query if tid == id]
    song1_features = np.array([f for _, p, f in current_triplet_list if p == 1])
    song2_features = np.array([f for _, p, f in current_triplet_list if p == 2])
    song3_features = np.array([f for _, p, f in current_triplet_list if p == 3])
    triplets.append(np.array([song1_features, song2_features, song3_features]))

triplets = np.array(triplets)

expected_triplets = np.array([
    [song1_analysis, song2_analysis, song3_analysis],
    [song1_1_analysis, song1_2_analysis, song1_3_analysis],
])
assert np.array_equal(expected_triplets, triplets)
