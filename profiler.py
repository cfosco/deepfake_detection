import json
import multiprocessing
import subprocess
from multiprocessing import Process

import gpustat
import torch
import torchvision.models as models


def get_power_reading():
    out = subprocess.Popen(['sudo', '/home/software/perftools/0.1/bin/satori-ipmitool'],
                           stdout=subprocess.PIPE)
    out = subprocess.Popen(['grep', 'Instantaneous power reading'], stdin=out.stdout, stdout=subprocess.PIPE)
    out = subprocess.run(['awk', '{print $4}'], stdin=out.stdout, stdout=subprocess.PIPE)
    if out.returncode == 0:
        return out.stdout.decode().strip()


class Profile:
    def __init__(self, outfile=None):
        self.outfile = outfile
        self.manager = multiprocessing.Manager()
        self.data = self.manager.list()

    @staticmethod
    def record_power(data):
        while True:
            q = gpustat.new_query().jsonify()
            q['query_time'] = q['query_time'].isoformat()
            data.append({
                **q,
                'power': get_power_reading(),
            })

    def __enter__(self):
        self.p = Process(target=self.record_power, args=(self.data,))
        self.p.start()
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.join()
        if self.outfile is not None:
            with open(self.outfile, 'w') as f:
                json.dump(list(self.data), f)
            print(f'Wrote output to {self.outfile}')
        return self


if __name__ == '__main__':

    model = models.densenet121(pretrained=True).cuda()
    x = torch.randn((64, 3, 224, 224), requires_grad=True).cuda()

    with Profile('test_profile.json') as prof:
        for i in range(100):
            out = model(x)
            if i % 10 == 0:
                print(i, out.shape)

    for d in prof.data:
        print(d)
        print()
