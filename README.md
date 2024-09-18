bliss metric learning
---------------------

This is a small local website to perform metric learning on your
[blissify-rs](https://github.com/Polochon-street/blissify-rs/) music library.

The website makes you do your own personal survey using your music library
on which songs are close to each other. Three by three, you will point the
odd-one-out, e.g. the song that is most dissimilar in the three songs that
will be presented to you.
Each "training triplet" (as it's called) will be stored in blissify's
database, and can be used to generate playlists that are closer to your
personal tastes.

📸 Pictures are better than a thousand words:
![2024-09-12-204506_735x370_scrot](https://github.com/user-attachments/assets/bdc3ece7-c469-4c37-b946-a70901352348)
![2024-09-12-204515_751x480_scrot](https://github.com/user-attachments/assets/1f698306-b9ba-424c-9b36-47a4c3dee40c)

TL;DR
-----

Install required python packages:

```
$ pip install -r requirements.txt
```

Start the local survey server:

```
$ python routes.py
```

Open your browser at http://127.0.0.1:5000, and complete the survey, answering as many rounds as you want (~200 is optimal, but 30-50 also works).

After that, stop the server by doing ctrl+c on the terminal.

Start the learning process (can take a few minutes):

```
$ python learn.py
```

Use the learned metric with blissify:

```
$ blissify playlist 300 --distance mahalanobis
```

Overall procedure
-----------------

(Note that this is heavily experimental, so do not expect it to directly have miraculous results!)

First, and with a working and fully analyzed [blissify-rs](https://github.com/Polochon-street/blissify-rs/) installation,
you would start the webserver by doing `python routes.py`.
You can then go to http://127.0.0.1:5000/ using your favorite web browser, and follow the instruction.
Rate as many songs as you want (~200 is optimal, but 50 is a very good start).

Once this is done, you can stop the webserver, and run `python learn.py`.
Note that it can take a while to do the learning, but once it is done, it should be stored directly
in blissify's config files.

You can then run `blissify playlist 300 --distance mahalanobis` to make a playlist using your newly trained
distance metric!

Technical details
-----------------

For the full details, see https://lelele.io/thesis.pdf, part 4: "Use of survey: metric learning".

bliss-rs represents each song as vectors with [NUMBER_FEATURES](https://docs.rs/bliss-audio/0.9.1/bliss_audio/constant.NUMBER_FEATURES.html) floating-point elements. By default, to compute distance between songs, bliss-rs uses the Euclidean distance: the distance between two songs A and B is $d(A, B) = \sqrt{(A - B)^{T}M(A - B)}$
