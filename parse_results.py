import os

def parse(PATH, MAX_STEP=100):
    files = list(os.walk(PATH))
    files = files[0][2]
    files = [f'{PATH}/{f}' for f in files if f.startswith('exp_init_run_worker')]
    tot = [0]*MAX_STEP
    for path in files:
        lines = []
        with open(path, 'r') as f:
            for line in f.read().splitlines():
                if line.startswith('('):
                    try:
                        line = eval(line)
                        lines.append(line)
                    except Exception:
                        continue
        for i, diff, time, acc in lines:
            if i >= MAX_STEP:
                break
            tot[i] += diff
    return tot

if not os.path.exists('results'):
    os.makedirs('results')

tup = parse('/path/to/your/results/dir')
with open(f'results/out.txt', 'w') as f:
    f.write(str(tup))