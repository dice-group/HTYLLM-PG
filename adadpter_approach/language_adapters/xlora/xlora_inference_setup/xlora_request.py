import requests, json

def query_xlora(prompt, max_tokens=500):
    url = "http://localhost:1234/v1/completions"
    data = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens
    }

    r = requests.post(url, headers={"Content-Type": "application/json"},
                      data=json.dumps(data))

    if r.ok:
        print(r.json()["choices"][0]["text"])
    else:
        print(r.status_code, r.text)

if __name__ == "__main__":
    query_xlora("How old is the universe?")
