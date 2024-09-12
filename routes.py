from xdg.BaseDirectory import xdg_data_home
from flask import Flask, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from markupsafe import Markup
from os import listdir
import sys
from os.path import isfile, join
from random import randint, choice
from sqlalchemy import desc, and_
from wtforms.fields import HiddenField, RadioField, SelectMultipleField
from wtforms import widgets
import sqlite3
import shutil
import os
from wtforms.validators import DataRequired
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Interval, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import glob
from collections import namedtuple

app = Flask(__name__)
app.secret_key = "AKEJkJDkjksjdJDZKJdkjzakjdKAJKJD"

Song = namedtuple("Song", "title artist album")

database_path = os.path.join(xdg_data_home, 'bliss-rs/songs.db')
con = sqlite3.connect(
    database_path, check_same_thread=False
)
cur = con.cursor()

songs = cur.execute(
    "select id, path, title, artist, album from song where analyzed = true"
)
all_songs = [
    (id, path, title, artist, album)
    for id, path, title, artist, album in songs.fetchall()
]
del cur


@app.route("/")
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return redirect(url_for("survey"))
    return render_template(
        "index.html",
        error=False,
    )


@app.route("/end_survey")
def end_survey():
    return render_template("end_survey.html")


def audio_url(filename):
    return Markup(
        '<audio class="player" controls controlsList="nodownload"><source src="{}" type="audio/mpeg"></audio>'
    ).format(filename)


class SurveyForm(FlaskForm):
    song1 = HiddenField()
    song2 = HiddenField()
    song3 = HiddenField()

    picked_song = RadioField("Pick One", validators=[DataRequired()], choices=[])


@app.route("/survey", methods=["GET", "POST"])
def survey():
    step = int(request.args.get("step", 0))

    if request.method == "POST":
        form = SurveyForm(request.form)
        if form.song1.data == form.picked_song.data:
            song1 = form.song2.data
            song2 = form.song3.data
            odd_one_out = form.song1.data
        elif form.song2.data == form.picked_song.data:
            song1 = form.song1.data
            song2 = form.song3.data
            odd_one_out = form.song2.data
        elif form.song3.data == form.picked_song.data:
            song1 = form.song1.data
            song2 = form.song2.data
            odd_one_out = form.song3.data
        cur = con.cursor()
        cur.execute(
            "insert into training_triplet (song_1_id, song_2_id, odd_one_out_id) values (?, ?, ?)",
            (song1, song2, odd_one_out),
        )
        con.commit()
        # Not pretty, not secure, must do bette
        files_to_delete = glob.glob("static/songs/*")
        for f in files_to_delete:
            os.remove(f)
    songs = [choice(all_songs), choice(all_songs), choice(all_songs)]
    for _, path, _, _, _ in songs:
        shutil.copyfile(path, f"static/songs/{os.path.basename(path)}")

    form = SurveyForm()
    form.song1.data = songs[0][0]
    form.song2.data = songs[1][0]
    form.song3.data = songs[2][0]

    form.picked_song.choices = [
        (
            id,
            Markup(
                f"{title} - {artist} - {album}"
                + audio_url(f"static/songs/{os.path.basename(path)}")
            ),
        )
        for (id, path, title, artist, album) in songs
    ]

    return render_template(
        "survey.html",
        form=form,
        step=step,
    )


def handle_survey(form):
    print(
        "Available choices were {}, {} and {}".format(
            form.song1, form.song2, form.song3
        )
    )
    print("Odd-one out is {}".format(form.picked_song.data.split()[0]))


if __name__ == "__main__":
    app.run(host="0.0.0.0")