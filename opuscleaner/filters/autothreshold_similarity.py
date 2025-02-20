#!/usr/bin/env python3


"""Requirements:
scikitlearn (https://scikit-learn.org)
scipy (https://scipy.org/)
comet (https://github.com/Unbabel/COMET)
"""

import sys
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy
from scipy import stats
from comet import download_model, load_from_checkpoint



def get_scores(stream, model, batch_size=20, verbose=None):
    lines = list()
    it = iter(stream)
    inputs = list()
    for line in it:
        src, tgt = line.rstrip("\r\n").split("\t")[:2]
        inputs.append({'src': src, 'mt': tgt})
        lines.append(line)

    model_output = model.predict(inputs, batch_size=batch_size, gpus=1)
    return np.array(model_output["scores"]), lines

def g(x, a, b):
    if x < a:
        return 0
    if x > b:
        return 1
    return (x - a)/ (b - a)

def weigthed_normals(xs, weights, means, vars, good=True, a=0.4, b=0.85):
    out = np.zeros(len(xs))
    if good:
        f = (lambda x: g(x, a, b))
    else:
        f = (lambda x: 1 - g(x, a, b))
    for w, mu, var in zip(weights, means, vars):
        out = out + f(mu) * w * stats.norm.pdf(xs, mu, np.sqrt(var))

    return out

def find_threshold(args, similarity_scores):

    assert len(similarity_scores) != 0
    y = similarity_scores[:5000]
    gm = GaussianMixture(n_components=args.numgaussians)
    gm.fit(y.reshape(-1, 1))

    xs = np.linspace(args.min, args.max, 100)
    f_good = weigthed_normals(xs, gm.weights_, gm.means_[:, 0], gm.covariances_[:, 0, 0], good=True, a=args.min, b=args.max) * (1 - args.prob)
    f_bad = weigthed_normals(xs, gm.weights_, gm.means_[:, 0], gm.covariances_[:, 0, 0], good=False, a=args.min, b=args.max) * args.prob
    deltas = f_good - f_bad

    threshold = xs[(deltas < 0).sum()]

    return threshold


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Filter a parallel dataset using COMETKIWI.")
    parser.add_argument("--verbose", action="store_true", help="Print tuning info")
    # parser.add_argument("--batch-size", type=int, help="LASER batch size")
    # parser.add_argument("--batch-latency", type=float, default=10.0, help="Tune batch size to process a batch every N seconds (defaults to 10s, ignored if --batch-size is given)")
    # parser.add_argument("--src-lang", type=str, required=True, help="Two-letter source language code (ISO 639-1)")
    # parser.add_argument("--tgt-lang", type=str, required=True, help="Two-letter target language code (ISO 639-1)")
    
    parser.add_argument("--min", type=float, help="Sentence alignment certainty lower bound.")
    parser.add_argument("--max", type=float, help="Sentence alignment certainty upper bound.")
    parser.add_argument("--numgaussians", type=int, help="Number of gaussian mixtures for density approximation.")
    parser.add_argument("--batchsize", type=int, default=20, help="Number of gaussian mixtures for density approximation.")
    parser.add_argument("--prob", type=float, help="Probability threshold of acceptance.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scores", action="store_true", help="Print scores instead of lines")
    group.add_argument("--threshold", type=float, help="Print scores instead of lines")

    args = parser.parse_args()

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    scores, lines = get_scores(sys.stdin, model, batch_size=args.batchsize, verbose=sys.stderr)

    threshold = find_threshold(args, scores)

    print(f"auto-threshold = {threshold:.3f}", file=sys.stderr)

    if not args.scores and args.threshold is None:
        print("Either use --threshold or --scores", file=sys.stderr)
    
    for line, score in zip(lines, scores):
        if score > threshold:
            sys.stdout.write(line)



if __name__ == "__main__":
    main()