#!/usr/bin/env python

from annoy import AnnoyIndex
from google.cloud import storage
import json
from os import mkdir, path, remove
from sparnn.config import DEBUG, VECTOR_SIZE
from sparnn import prep
from time import time
from typing import Dict, List

START_TIME = int(time())

num_rows = 0
num_files = 0

max_user = 0.0
min_user = 0.0
max_track = 0.0
min_track = 0.0

uris: Dict[str, int] = {}
uri_index: List[str] = []
uri_tree = AnnoyIndex(VECTOR_SIZE, "angular")


def log(message: str) -> str:
    hours = "{:.4f}".format((int(time()) - START_TIME) * 1.0 / (60 * 60))
    return f"{hours}h: {message}"


if __name__ == "__main__":
    print(f"DEBUG: ({DEBUG})")
    if not path.exists(f"./data"):
        mkdir(f"./data")
    if not path.exists(f"./data/prep"):
        mkdir(f"./data/prep")

    blobs = [blob for blob in storage.Client().list_blobs("sparnn")]
    with open("./data/prep/examples.csv", mode="w", encoding="utf-8") as f:
        for blob in blobs:
            if DEBUG and num_files >= 1:
                break
            filename = prep.download(blob)
            if filename is not None:
                session = prep.parse(filename)
                for s in session:
                    user_uri = f"spotify:userid:{s.user_id}"
                    indices = []
                    if user_uri in uris.keys():
                        indices.append(uris[user_uri])
                    else:
                        max_user = max(s.user_vector)
                        min_user = min(s.user_vector)

                        user_index = len(uri_index)
                        uri_index.append(user_uri)
                        indices.append(user_index)
                        uri_tree.add_item(user_index, s.user_vector)
                        uris[user_uri] = user_index
                    for p in s.prior_plays:
                        if p.uri in uris.keys():
                            indices.append(uris[p.uri])
                        else:
                            max_track = max(p.vector)
                            min_track = min(p.vector)

                            prior_i = len(uri_index)
                            uri_index.append(p.uri)
                            indices.append(prior_i)
                            uri_tree.add_item(prior_i, p.vector)
                            uris[p.uri] = prior_i
                    if s.next_play.uri in uris.keys():
                        next_play_index = uris[s.next_play.uri]
                    else:
                        max_track = max(s.next_play.vector)
                        min_track = min(s.next_play.vector)

                        next_play_index = len(uri_index)
                        uri_index.append(s.next_play.uri)
                        indices.append(next_play_index)
                        uri_tree.add_item(next_play_index, s.next_play.vector)
                        uris[s.next_play.uri] = next_play_index
                    row = ",".join(map(str, indices))
                    f.write(f"{row}\n")
                    num_rows += 1
                    if num_rows % 10000 == 0:
                        print(log(f"wrote {num_rows} rows"))
                num_files += 1
                remove(filename)
                if num_files % 10 == 0:
                    print(log(f"read {num_files} files"))

    uri_tree.build(1000)
    uri_tree.save("./data/prep/uri.ann")

    with open("./data/prep/uri.txt", mode="w", encoding="utf-8") as f:
        for uri in uri_index:
            f.write(f"{uri}\n")

    with open("./data/prep/stats.json", mode="w", encoding="utf-8") as f:
        stats = {
            "uri_keys": len(uris.keys()),
            "uri_index": len(uri_index),
            "annoy_items": uri_tree.get_n_items(),
            "max_user": max_user,
            "min_user": min_user,
            "max_track": max_track,
            "min_track": min_track,
        }
        f.write(f"{json.dumps(stats)}\n")