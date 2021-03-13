from .config import DEBUG, VECTOR_SIZE
import gzip
import json
from random import randint
from os import mkdir, path
from typing import Generator, List, NamedTuple, Optional, Tuple

BLOB_PREFIX = "user_play/"
PRIOR_PLAYS = 5
SESSION_SPLIT_SECS = 30 * 60  # session defined as 30 minutes of inactivity

if not path.exists(f"./data"):
    mkdir(f"./data")
if not path.exists(f"./data/{BLOB_PREFIX}"):
    mkdir(f"./data/{BLOB_PREFIX}")


class TrackPlay(NamedTuple):
    uri: str
    first_play_ms: float
    vector: List[float]


class Session(NamedTuple):
    user_id: str
    user_vector: List[float]
    prior_plays: List[TrackPlay]
    next_play: TrackPlay


def download(blob) -> Optional[str]:
    if not blob.name.startswith(BLOB_PREFIX):
        return None
    local_filename = f"./data/{blob.name}"
    if not path.isfile(local_filename):
        blob.download_to_filename(local_filename)
    return local_filename


def find_session(plays) -> Optional[Tuple[List[TrackPlay], TrackPlay]]:
    if len(plays) <= PRIOR_PLAYS:
        return None
    midpoint = randint(PRIOR_PLAYS, len(plays))
    for i, p in enumerate(plays[midpoint:]):
        index = midpoint + i
        start = plays[index - 1]["first_play_ms"]
        end = p["first_play_ms"]
        if end - start >= SESSION_SPLIT_SECS:
            prior_plays = []
            for pp in plays[index - PRIOR_PLAYS : index]:
                track_size = len(pp["track_vector"])
                if track_size != VECTOR_SIZE:
                    track_uri = pp["track_uri"]
                    print(
                        f"track:({track_uri}) vector was ({track_size}) expected ({VECTOR_SIZE})"
                    )
                    return None
                prior_plays.append(
                    TrackPlay(
                        uri=pp["track_uri"],
                        first_play_ms=pp["first_play_ms"],
                        vector=pp["track_vector"],
                    )
                )
            track_size = len(p["track_vector"])
            if track_size != VECTOR_SIZE:
                track_uri = p["track_uri"]
                print(
                    f"track:({track_uri}) vector was ({track_size}) expected ({VECTOR_SIZE})"
                )
                return None
            return (
                prior_plays,
                TrackPlay(
                    uri=p["track_uri"],
                    first_play_ms=p["first_play_ms"],
                    vector=p["track_vector"],
                ),
            )
    return None


def parse(filename) -> Generator[Session, None, None]:
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_size = len(row["user_vector"])
            if user_size != VECTOR_SIZE:
                user_id = row["user_id"]
                print(
                    f"userid:({user_id}) vector was ({user_size}) expected ({VECTOR_SIZE})"
                )
                continue
            session = find_session(row["plays"])
            if session is None:
                continue
            prior_plays, next_play = session
            yield Session(
                user_id=row["user_id"],
                user_vector=row["user_vector"],
                prior_plays=prior_plays,
                next_play=next_play,
            )
    return None
