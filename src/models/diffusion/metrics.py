from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(input_img_size=(3,28,28))


class Metrics():
    def __init__(self, device):
        self.fid = FrechetInceptionDistance(input_img_size=(3,28,28))

    def _calculate_fid(self, real_pcs, gen_pcs):
        real_pcs = real_pcs.unsqueeze(1)
        real_pcs = real_pcs.repeat(1, 3, 1, 1).cpu()
        gen_pcs = gen_pcs.unsqueeze(1)
        gen_pcs = gen_pcs.repeat(1, 3, 1, 1).cpu()
        self.fid.update(real_pcs, real=True)
        self.fid.update(gen_pcs, real=False)
        return self.fid.compute()

    def compute_all_metrics(self, real_pcs, gen_pcs):
        return {"fid": self._calculate_fid(real_pcs=real_pcs, gen_pcs=gen_pcs)}