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

Overall procedure
-----------------

(Note that this is heavily experimental, so do not expect it to directly have miraculous results!)

First, and with a working and fully analyzed [blissify-rs](https://github.com/Polochon-street/blissify-rs/) installation,
you would start the webserver by doing `python routes.py`.
You can then go to http://127.0.0.1:5000/ using your favorite web browser, and follow the instruction.
Rate as many songs as you want (50-100 is a very good start).

Once this is done, you can stop the webserver, and run `python make_distance_matrix.py`.
Note that it can take a while to do the learning, but once it is done, it should be stored directly
in blissify's config files.

You can then run `blissify playlist 30 --distance mahalanobis` to make a playlist using your newly trained
distance metric!

TODO add a section "technical details"
