from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()
sys.path.insert(0, "/vast/home/m/mgjacob/PARCC/scripts/theseus")
import theseus as th
from theseus.geometry.point_types import Point3 
from theseus.geometry.se3 import SE3
import time 

class TheseusRelativeSE3Solver:
    def __init__(self, detach_scale=True, eps=1e-8):
        self.detach_scale = detach_scale
        self.eps = eps
        self._cache = {}  # keyed by (N, device, dtype)

    def _build(self, N, device, dtype, optim_steps, step_size):
        # pairs / indices (cache on device)
        pairs = torch.tensor(list(combinations(range(N), 2)), device=device, dtype=torch.long)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        E = pairs.shape[0]

        # Theseus optim vars with fixed names
        opt_poses = [th.SE3(name=f"T{i}") for i in range(N)]

        # aux var with fixed shape [1,E,3,4]
        gt_rel_var = th.Variable(
            torch.zeros(1, E, 3, 4, device=device, dtype=dtype),
            name="Tgt_rel",
        )

        detach_scale = self.detach_scale
        eps = self.eps

        def err_fn(optim_vars, aux_vars):
            (Tgt_rel_,) = aux_vars
            Tgt_rel = Tgt_rel_.tensor.view(E, 3, 4)

            # Stack predicted absolute poses: [N,3,4]
            Tpred = torch.cat([v.tensor for v in optim_vars], dim=0)

            Ti = th.SE3(tensor=Tpred[i_idx], disable_checks=True)  # [E,3,4]
            Tj = th.SE3(tensor=Tpred[j_idx], disable_checks=True)  # [E,3,4]
            Tij_pred = Ti.inverse().compose(Tj)                    # [E,3,4]

            Tij_gt = th.SE3(tensor=Tgt_rel, disable_checks=True)

            tp = Tij_pred.tensor[:, :, 3]  # [E,3]
            tg = Tij_gt.tensor[:, :, 3]    # [E,3]

            num = (tp * tg).sum()
            den = (tp * tp).sum() + eps
            s_star = num / den
            if detach_scale:
                s_star = s_star.detach()

            Rpred = Tij_pred.tensor[:, :, :3]               # [E,3,3]
            tpred = Tij_pred.tensor[:, :, 3:4] * s_star     # [E,3,1]
            Tij_pred_scaled = th.SE3(
                tensor=torch.cat([Rpred, tpred], dim=2),
                disable_checks=True
            )

            delta = Tij_pred_scaled.inverse().compose(Tij_gt)
            r6 = delta.log_map()                            # [E,6]
            return r6.view(1, 6 * E)

        cost_fn = th.AutoDiffCostFunction(
            tuple(opt_poses),
            err_fn,
            6 * E,
            aux_vars=(gt_rel_var,),
            name="relative_se3_cost",
        )

        obj = th.Objective()
        obj.add(cost_fn)
        obj = obj.to(device)

        optimizer = th.LevenbergMarquardt(
            obj,
            max_iterations=optim_steps,
            step_size=step_size,
        )

        layer = th.TheseusLayer(optimizer)

        return {
            "N": N, "E": E,
            "pairs": pairs,
            "opt_poses": opt_poses,
            "gt_rel_var": gt_rel_var,
            "layer": layer,
            "optimizer": optimizer,
        }

    def solve(self, R_t, T_t, gt_poses_34, optim_steps=15, step_size=1.0, verbose=False):
        assert R_t.shape[0] == 1 and T_t.shape[0] == 1
        device, dtype = R_t.device, R_t.dtype
        N = R_t.shape[1]

        key = (N, device, dtype, optim_steps, step_size)
        if key not in self._cache:
            self._cache[key] = self._build(N, device, dtype, optim_steps, step_size)

        cache = self._cache[key]
        pairs = cache["pairs"]
        E = cache["E"]

        # build GT relative edges each call (unless GT fixed)
        with torch.no_grad():
            gt_rel = []
            for ii, jj in pairs.tolist():
                Ti = th.SE3(tensor=gt_poses_34[:, ii], disable_checks=True)
                Tj = th.SE3(tensor=gt_poses_34[:, jj], disable_checks=True)
                gt_rel.append(Ti.inverse().compose(Tj).tensor)  # [1,3,4]
            gt_rel = torch.cat(gt_rel, dim=0).view(1, E, 3, 4).contiguous()

        # inputs: only tensors
        inputs = {f"T{i}": se3_from_rot_trans(R_t[:, i], T_t[:, i])[..., :3, :4].contiguous()
                  for i in range(N)}
        inputs["Tgt_rel"] = gt_rel

        updated, info = cache["layer"].forward(
            inputs,
            optimizer_kwargs={"verbose": verbose, "track_best_solution": True},
        )

        best = info.best_solution
        T_out = torch.cat([best[f"T{i}"] for i in range(N)], dim=0)  # [N,3,4]
        R_out = T_out[:, :, :3].unsqueeze(0)
        t_out = T_out[:, :, 3].unsqueeze(0)
        return R_out, t_out, info
