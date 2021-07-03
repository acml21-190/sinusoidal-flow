import torch
import torch.nn as nn
from torch.distributions import Normal

from project.models.affine import AffineTransformer, IdentityTransformer, validate_cond_var, MaskedLinear, \
    FlipTransformer
from project.models.normaliser import DiffPointwiseActNorm, PointwiseActNorm
from project.models.pixelcnn import PixelCNN


class IndependentConditioner(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int):
        super(IndependentConditioner, self).__init__()

        self.in_proj_weight = nn.Parameter(torch.empty(c, h, w, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty_like(self.in_proj_weight))
        self.out_proj_weight = nn.Parameter(torch.empty_like(self.in_proj_weight))
        self.out_proj_bias = nn.Parameter(torch.empty(c, h, w))
        self.residual_weight = nn.Parameter(torch.empty_like(self.out_proj_bias))

        self.softplus = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.in_proj_weight, -3, 3)
        nn.init.uniform_(self.in_proj_bias, -3, 3)
        nn.init.zeros_(self.out_proj_weight)
        nn.init.zeros_(self.out_proj_bias)
        nn.init.zeros_(self.residual_weight)

    def forward(self, x):
        in_proj_weight = self.softplus(self.in_proj_weight)
        return in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.residual_weight


class MLPConditioner(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int, in_dim=1, hid_dims=tuple(), **kwargs):
        super(MLPConditioner, self).__init__()

        assert isinstance(hid_dims, tuple), "hid_dims must be a tuple. "

        flow_dim = c * h * w
        out_dim = 3 * flow_dim * embed_dim + 2 * flow_dim
        hid_dims += (out_dim,)

        mlp = nn.ModuleList([nn.Linear(in_features=in_dim, out_features=hid_dims[0])])
        for i in range(1, len(hid_dims)):
            mlp.extend([nn.ReLU(),
                        nn.Linear(in_features=hid_dims[i - 1], out_features=hid_dims[i])])
        self.mlp = nn.Sequential(*mlp)
        self.chw = c, h, w
        self.flow_dim = flow_dim
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):  # (*, D)
        batch_dims = x.size()[:-1]
        out = self.mlp(x)

        weights = out[..., :-2 * self.flow_dim].reshape(*batch_dims, *self.chw, 3, -1)  # (*, C, H, W, 3, E)
        in_proj_weight, in_proj_bias, out_proj_weight = torch.unbind(weights, dim=-2)
        in_proj_weight = self.softplus(in_proj_weight)

        out_proj_bias, residual_weight = torch.unbind(out[..., -2 * self.flow_dim:].reshape(*batch_dims, *self.chw, 2),
                                                      dim=-1)  # (*, C, H, W)

        return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight


class MaskedMLPConditioner(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int, hid_dims=(100,), use_tanh=False, **kwargs):
        super(MaskedMLPConditioner, self).__init__()

        assert isinstance(hid_dims, tuple), "hid_dims must be a tuple. "

        flow_dim = c * h * w
        out_dim = 3 * flow_dim * embed_dim + 2 * flow_dim
        hid_dims += (out_dim,)

        mlp = nn.ModuleList([MaskedLinear(flow_dim, in_features=flow_dim, out_features=hid_dims[0], mask_type="input")])
        for i in range(1, len(hid_dims)):
            mask_type = None if i < len(hid_dims) - 1 else "output"
            mlp.extend([
                nn.Tanh() if use_tanh else nn.ReLU(),
                MaskedLinear(flow_dim, in_features=hid_dims[i - 1], out_features=hid_dims[i], mask_type=mask_type)
            ])
        self.mlp = nn.Sequential(*mlp)
        self.chw = c, h, w
        self.flow_dim = flow_dim
        self.softplus = nn.Softplus()
        self.norm = PointwiseActNorm(c, h, w)

    def forward(self, x: torch.Tensor):  # (*, C, H, W)
        x = torch.flatten(self.norm(x), start_dim=-3)  # (*, CHW)
        batch_dims = x.size()[:-1]
        out = self.mlp(x)

        weights = out[..., :-2 * self.flow_dim].reshape(*batch_dims, -1, self.flow_dim).transpose(-2, -1)
        weights = weights.reshape(*batch_dims, *self.chw, 3, -1)  # (*, C, H, W, 3, E)
        in_proj_weight, in_proj_bias, out_proj_weight = torch.unbind(weights, dim=-2)
        in_proj_weight = self.softplus(in_proj_weight)

        out_proj_bias, residual_weight = torch.unbind(out[..., -2 * self.flow_dim:].reshape(*batch_dims, 2, *self.chw),
                                                      dim=-4)  # (*, C, H, W)

        return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight


class PixelCNNConditioner(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int, num_fmaps=16, num_blocks=4, **kwargs):
        super(PixelCNNConditioner, self).__init__()

        self.pixel_cnn = PixelCNN(c, 5 * c, num_fmaps, num_blocks, h, w, normaliser="actnorm")

        self.in_proj_weight_embed = nn.Linear(in_features=1, out_features=embed_dim)
        self.in_proj_bias_embed = nn.Linear(in_features=1, out_features=embed_dim)
        self.out_proj_weight_embed = nn.Linear(in_features=1, out_features=embed_dim)

        self.softplus = nn.Softplus()
        self.norm = PointwiseActNorm(c, h, w)

    def forward(self, x: torch.Tensor):  # (N, C, H, W)
        x = self.norm(x)
        out = self.pixel_cnn(x)  # (N, 5C, H, W)

        in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight = torch.chunk(out, 5, dim=-3)

        in_proj_weight = self.softplus(self.in_proj_weight_embed(in_proj_weight.unsqueeze(dim=-1)))  # (N, C, H, W, E)
        in_proj_bias = self.in_proj_bias_embed(in_proj_bias.unsqueeze(dim=-1))  # (N, C, H, W, E)
        out_proj_weight = self.out_proj_weight_embed(out_proj_weight.unsqueeze(dim=-1))  # (N, C, H, W, E)

        return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight


class LinearAttentionConditioner(nn.Module):

    def __init__(self, c, h, w, embed_dim, multiplier=1, **kwargs):
        super(LinearAttentionConditioner, self).__init__()

        self.num_heads = 5
        self.embed_weight = nn.Parameter(torch.empty(self.num_heads * 3, c, h, w, multiplier * embed_dim))
        self.embed_bias = nn.Parameter(torch.empty_like(self.embed_weight))
        self.ind_cond = IndependentConditioner(1, 1, 1, multiplier * embed_dim)

        self.softplus = nn.Softplus()

        self.fc = nn.Sequential(
            nn.Linear(in_features=multiplier * embed_dim, out_features=2 * embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=2 * embed_dim, out_features=embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embed_weight)
        nn.init.xavier_uniform_(self.embed_bias)

    def forward(self, x):
        n, c, h, w = x.size()

        x = x.unsqueeze(dim=1).unsqueeze(dim=-1)  # (N, 1, C, H, W, 1)

        q, k, v = torch.chunk(self.embed_weight * x + self.embed_bias, 3, dim=1)
        q = q.reshape(q.size(0), self.num_heads, -1, q.size(-1))
        k = k.reshape(k.size(0), self.num_heads, -1, k.size(-1))
        v = v.reshape(v.size(0), self.num_heads, -1, v.size(-1))

        q, k = self.softplus(q), self.softplus(k)
        s = torch.einsum("nhsi,nhsj->nhsij", k, v)  # batch outer product
        s = torch.cumsum(s, dim=2)  # (N, h, S, E, E)
        z = torch.cumsum(k, dim=2)

        numer = torch.einsum("nhsi,nhsij->nhsj", q, s)  # (N, h, S, E)
        denom = torch.einsum("nhsi,nhsi->nhs", q, z).unsqueeze(dim=-1)  # (N, h, S, 1)
        params = numer / denom  # (N, h, S, E)

        inner_params, outer_params = params[:, :-2], params[:, -2:]  # (N, 3, S, E), (N, 2, S, E)
        outer_params = outer_params.sum(dim=-1, keepdims=True)  # (N, 2, S, 1)

        ind_params = self.ind_cond(x)
        ind_inner_params, ind_outer_params = ind_params[:3], ind_params[3:]
        ind_inner_params = torch.stack(ind_inner_params, dim=0).reshape(3, 1, -1).expand(n, -1, -1, -1)  # (N, 3, 1, E)
        ind_outer_params = torch.stack(ind_outer_params, dim=0).reshape(2, 1, -1).expand(n, -1, -1, -1)  # (N, 2, 1, 1)

        # Strictly autoregressive
        inner_params = torch.cat([ind_inner_params, inner_params[:, :, :-1]], dim=2)  # (N, 3, S, E)
        outer_params = torch.cat([ind_outer_params, outer_params[:, :, :-1]], dim=2)  # (N, 2, S, 1)

        inner_params = self.fc(inner_params)

        in_proj_weight, in_proj_bias, out_proj_weight = torch.unbind(inner_params, dim=1)
        in_proj_weight = self.softplus(in_proj_weight.reshape(n, c, h, w, -1))
        in_proj_bias = in_proj_bias.reshape(n, c, h, w, -1)
        out_proj_weight = out_proj_weight.reshape(n, c, h, w, -1)

        out_proj_bias, residual_weight = torch.unbind(outer_params, dim=1)
        out_proj_bias = out_proj_bias.reshape(n, c, h, w)
        residual_weight = residual_weight.reshape(n, c, h, w)

        return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight


class SinusoidalTransformer(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int, conditioner: str, **kwargs):
        super(SinusoidalTransformer, self).__init__()

        if conditioner == "ind":
            self.conditioner = IndependentConditioner(c, h, w, embed_dim)
        elif conditioner == "mlp":
            self.conditioner = MLPConditioner(c, h, w, embed_dim, **kwargs)
        elif conditioner == "msk":
            self.conditioner = MaskedMLPConditioner(c, h, w, embed_dim, **kwargs)
        elif conditioner == "cnn":
            self.conditioner = PixelCNNConditioner(c, h, w, embed_dim, **kwargs)
        elif conditioner == "atn":
            self.conditioner = LinearAttentionConditioner(c, h, w, embed_dim, **kwargs)
        else:
            self.conditioner = None
            raise ValueError("Invalid conditioner. ")

        self._inverting = False

    def forward(self, x: torch.Tensor, cond_var: torch.Tensor = None):
        """
        :param x: input variable of shape (*, C, H, W)
        :param cond_var: the conditional variable of shape (*, D) or x itself
        :return: the transformed variable of shape (*, C, H, W) and the log determinant of shape (*, )
        """
        assert validate_cond_var(x, cond_var), "Your cond_var is not compatible in shape with x. "

        cond_var = x if cond_var is None else cond_var

        in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight = self.conditioner(cond_var)
        mix_weights = torch.softmax(out_proj_weight, dim=-1)
        residual_weight = torch.tanh(residual_weight)

        u = in_proj_weight * x.unsqueeze(dim=-1) + in_proj_bias
        y = (torch.sin(2 * in_proj_bias) - torch.sin(2 * u)) / (2 * in_proj_weight)
        y = x + residual_weight * torch.sum(mix_weights * y, dim=-1) + out_proj_bias

        log_dets = None
        if not self._inverting:
            dy_dx = -residual_weight * torch.sum(mix_weights * torch.cos(2 * u), dim=-1)  # (*, C, H, W)
            log_dets = torch.log1p(dy_dx).sum(dim=(-3, -2, -1))

        return y, log_dets

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=100):
        assert not self.training, "inv_transform() cannot be called during training. "
        assert validate_cond_var(z, cond_var), "Your cond_var is not compatible in shape with z. "

        self._inverting = True

        num_iter = 1
        x_k = z
        x_kp1 = z - (self(x_k, cond_var)[0] - x_k)
        while num_iter < max_iter and not torch.allclose(x_kp1, x_k, rtol=rtol, atol=atol):
            x_k = x_kp1
            x_kp1 = z - (self(x_k, cond_var)[0] - x_k)
            num_iter += 1

        self._inverting = False

        return x_kp1


class SinusoidalFlow(nn.Module):
    def __init__(self, c: int, h: int, w: int, embed_dim: int,
                 conditioner: str, num_layers: int, affine=False, **kwargs):
        super(SinusoidalFlow, self).__init__()

        self.transformers = nn.ModuleList([SinusoidalTransformer(c, h, w, embed_dim, conditioner, **kwargs)])
        for _ in range(num_layers - 1):
            self.transformers.extend([
                AffineTransformer(c, h, w, conditioner="ind") if affine else IdentityTransformer(),
                FlipTransformer() if conditioner != "ind" else IdentityTransformer(),
                SinusoidalTransformer(c, h, w, embed_dim, conditioner, **kwargs)
            ])

        self.norm = DiffPointwiseActNorm(c, h, w)
        self.base_dist = Normal(0.0, 1.0)
        self.chw = c, h, w

    def forward(self, x, cond_var=None):
        sum_log_dets = []

        for layer in self.transformers:
            x, log_dets = layer(x, cond_var)
            sum_log_dets.append(log_dets)

        z, log_dets = self.norm(x)
        sum_log_dets.append(log_dets)

        sum_log_dens = torch.sum(self.base_dist.log_prob(z), dim=(-3, -2, -1))
        sum_log_dets = torch.stack(sum_log_dets, dim=0).sum(dim=0)
        log_joint_dens = sum_log_dens + sum_log_dets

        return z, log_joint_dens

    @torch.no_grad()
    def inv_transform(self, z, cond_var=None, rtol=1e-5, atol=1e-8, max_iter=100):
        assert not self.training, "inv_transform() cannot be called during training. "

        z = self.norm.inv_transform(z, cond_var)
        for layer in reversed(self.transformers):
            z = layer.inv_transform(z, cond_var, rtol, atol, max_iter)
        return z

    @torch.no_grad()
    def sample(self, sample_size, temp=1.0, cond_var=None, batch_size=128, rtol=1e-10, atol=1e-16, max_iter=100):
        assert not self.training, "sample() cannot be called during training. "

        device = self.norm.log_std.device
        self.double()

        # samples = []
        # while len(samples) < sample_size // batch_size + 1:
        #     size = min(batch_size, sample_size - len(samples) * batch_size)
        #     z = self.base_dist.sample((size, *self.chw)).to(device, torch.double)
        #     y = self.inv_transform(z, cond_var, rtol, atol, max_iter)
        #     samples.append(y.cpu().float())
        # samples = torch.cat(samples, dim=0)

        samples = []
        z = self.base_dist.sample((sample_size, *self.chw)) * temp
        for batch in torch.split(z, batch_size, dim=0):
            batch = batch.to(device, torch.double)
            y = self.inv_transform(batch, cond_var, rtol, atol, max_iter)
            samples.append(y.cpu().float())
        samples = torch.cat(samples, dim=0)

        self.float()
        return samples


class SinusoidalFlowRegressor(SinusoidalFlow):
    def __init__(self, embed_dim, conditioner, num_layers, affine=True, **kwargs):
        super(SinusoidalFlowRegressor, self).__init__(1, 1, 1, embed_dim, conditioner, num_layers, affine, **kwargs)

        # self.transformers.append(AffineTransformer(*self.chw, conditioner="mlp", **kwargs))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the independent variable of shape (N, D)
        :return: the predicted value of shape (N, 1)
        """
        assert not self.training, "predict() cannot be called during training. "

        modes = self.calc_modes(x)
        top_modes = []
        for item in modes:
            if item:
                top_modes.append(item[0][[0]])
            else:
                top_modes.append(torch.tensor([float("nan")]))
        top_modes = torch.stack(top_modes)
        return top_modes

    @torch.no_grad()
    def calc_mean(self, x, sample_size=10000, batch_size=128, rtol=1e-10, atol=1e-16, max_iter=100) -> torch.Tensor:
        """
        Calculates the mean of the conditional distribution p(y | x)

        :param x: the conditional variable of shape (N, D)
        :param sample_size: the size of the sample used
        :param batch_size: the number of samples used for calculation at one time
        :param rtol: relative tolerance
        :param atol: absolute tolerance
        :param max_iter: the maximum number of iterations before the algorithm should stop
        :return: the estimated mean of shape (N, 1)
        """
        assert not self.training, "calc_mean() cannot be called during training. "

        # use double to preserve precision in the intermediate results
        x = x.double()
        self.double()

        prob_delta = 1 / sample_size
        z = self.base_dist.icdf(x.new_full((sample_size - 1,), prob_delta).cumsum(dim=-1))

        even_z = z[::2].expand(*tuple(reversed(self.chw)), x.size(0), -1).T  # (S // 2, N, C, H, W)
        odd_z = z[1::2].expand(*tuple(reversed(self.chw)), x.size(0), -1).T

        even_sums = []
        for batch in torch.split(even_z, batch_size, dim=0):
            even_sums.append(self.inv_transform(batch, cond_var=x, rtol=rtol, atol=atol, max_iter=max_iter).sum(dim=0))
        even_sums = torch.stack(even_sums, dim=0).sum(dim=0)

        odd_sums = []
        for batch in torch.split(odd_z, batch_size, dim=0):
            odd_sums.append(self.inv_transform(batch, cond_var=x, rtol=rtol, atol=atol, max_iter=max_iter).sum(dim=0))
        odd_sums = torch.stack(odd_sums, dim=0).sum(dim=0)

        # Simpson’s Rule
        y = 4 * even_sums + 2 * odd_sums
        y += self.inv_transform(even_z[[0, -1]], cond_var=x, rtol=rtol, atol=atol, max_iter=max_iter).sum(dim=0)
        y *= prob_delta / 3

        self.float()
        y = y.float().reshape(-1, 1)

        return y

    @torch.no_grad()
    def calc_modes(self, x: torch.Tensor, step=1e-2, batch_size=128, low_prob=1e-3, tol=1e-5, max_iter=100):
        """
        Computes the modes of the conditional distribution p(y | x)

        :param x: the conditional variable of shape (N, D)
        :param step: the increment between each two sample points
        :param batch_size: the number of sample points used for calculation at one time
        :param low_prob: the lowest probability to start with
        :param tol: tolerance for Newton's method
        :param max_iter: the maximum number of iterations before the algorithm should stop
        :return: an N-long list of tuples with two tensors representing the modes and the density at each mode
        """
        assert not self.training, "calc_modes() cannot be called during training. "

        # Freeze the model
        param_grad = dict()
        for name, param in self.named_parameters():
            param_grad[name] = param.requires_grad
            param.requires_grad = False

        modes = []
        for curr_x in torch.unbind(x, dim=0):  # (D, )
            # Determine the lower and upper bounds of search
            z = curr_x.new_tensor([low_prob, 1 - low_prob]).reshape(2, *self.chw)
            bounds = self.inv_transform(self.base_dist.icdf(z), cond_var=curr_x.expand(2, -1))
            low, high = bounds[0].item(), bounds[1].item()
            samples = torch.linspace(low, high, int((high - low) / step)).reshape(-1, *self.chw)

            candidates = []
            for y in torch.split(samples, batch_size, dim=0):
                y = y.to(curr_x).requires_grad_(True)
                num_iter, diff = 0, torch.ones_like(y)
                while num_iter < max_iter and torch.gt(diff, tol).any():
                    with torch.enable_grad():
                        log_prob = self(y, cond_var=curr_x.expand(y.size(0), -1))[-1]
                        dlogprob_dy = torch.autograd.grad(-log_prob.sum(), y, create_graph=True)[0]
                        d2logprob_dy2 = torch.autograd.grad(dlogprob_dy.sum(), y)[0]

                    yp = y - dlogprob_dy / d2logprob_dy2  # Newton's step
                    diff = torch.abs(yp - y)
                    y.data = yp
                    num_iter += 1

                candidates.append(y[(low <= y) & (y <= high)].detach().cpu())  # also removes NaN

            y = torch.cat(candidates, dim=-1)

            # If no stationary points are found
            if y.numel() == 0:
                # Evaluate log-likelihood at the two bounds
                bounds_log_prob = self(bounds, cond_var=curr_x.expand(2, -1))[-1]
                bounds, bounds_log_prob = bounds[~bounds_log_prob.isnan()], bounds_log_prob[~bounds_log_prob.isnan()]
                if bounds.numel() == 0:  # no answers are found
                    modes.append(tuple())
                else:
                    bigger_index = bounds_log_prob.argmax().item()
                    modes.append((bounds[bigger_index], bounds_log_prob[bigger_index]))
                continue

            # y = torch.unique(torch.round(y / tol) * tol).reshape(-1, *self.chw)
            y = torch.sort(y)[0]
            y = torch.cat([y[[0]], y[1:][~torch.isclose(y[:-1], y[1:], rtol=step / 10)]]).reshape(-1, *self.chw)

            y = y.to(curr_x).requires_grad_(True)

            with torch.enable_grad():
                log_prob = self(y, cond_var=curr_x.expand(y.size(0), -1))[-1]
                dlogprob_dy = torch.autograd.grad(log_prob.sum(), y, create_graph=True)[0]
                d2logprob_dy2 = torch.autograd.grad(dlogprob_dy.sum(), y)[0]

            # Use the second derivative test to select maxima
            fst_ord_cond = torch.isclose(dlogprob_dy, torch.zeros_like(dlogprob_dy), atol=1e-5)
            sec_ord_cond = torch.lt(d2logprob_dy2, 0.0)
            is_maximum = torch.logical_and(fst_ord_cond, sec_ord_cond)

            y = y[is_maximum].detach().cpu()
            log_prob = log_prob.reshape(-1, *self.chw)[is_maximum].detach().cpu()
            indices = torch.argsort(log_prob, descending=True)
            y, log_prob = y[indices], log_prob[indices]
            modes.append((y, log_prob))

        # Unfreeze the model
        for name, param in self.named_parameters():
            param.requires_grad = param_grad[name]

        return modes
