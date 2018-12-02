import argparse
import requests
import sys

parser = argparse.ArgumentParser(description='Compute most similar wikipedia articles')
parser.add_argument('-m', type=str, help="Name of saved model.")
parser.add_argument('-u', type=str, help="URL of API.")
parser.add_argument('-n', type=int, default=50, help="Number of similar documents to find.")
parser.add_argument('--method', type=str, default="annoy", choices=['annoy', 'brute', 'kdtree'], help="Method for performing NNS.")
parser.add_argument('-t', type=str, help="The text to query for")


def _main(text, modelname, url, n, method):
    if modelname is not None:
        from doc2vecmodel import Doc2VecModel
        d2v = Doc2VecModel(modelname)
        titles, scores =  d2v.infer(text, n)

    elif url is not None:
        response = requests.post(url, data={'text': text, 'n': n, 'method': method})
        if response.status_code != 200:
            raise Exception(f"Error recived status code: {response.status_code}")
        json_obj = response.json()
        titles = json_obj['titles']
        scores = json_obj['scores']
    else:
        raise Exception("You must supply either modelname or ip")

    print(f"---------------------------- TOP {n} similar texts ----------------------------")
    print('{:<60}  {:<60}'.format("TITLE", "SIMILARITY SCORE"))
    for i in range(n):
        print('{:<60}  {:<60}'.format(str(titles[i]), str(scores[i])))

if __name__ == "__main__":
    args = parser.parse_args()
    if args.t is not None:
        _main(args.t, args.m, args.u, args.n, args.method)
    else:
        text = ' '.join(sys.stdin.readlines())
        _main(text, args.m, args.u, args.n, args.method)