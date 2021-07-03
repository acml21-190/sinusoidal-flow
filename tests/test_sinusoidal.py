import torch
import unittest
from torch.autograd.functional import jacobian

from project.models.depthwise import DepthwiseSinusoidalTransformer
from project.models.sinusoidal import SinusoidalTransformer, SinusoidalFlow, MLPConditioner, MaskedMLPConditioner, \
    LinearAttentionConditioner


class MLPConditionerTestCase(unittest.TestCase):

    def test_mlp(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        hid_dims = (8, 4)
        m, n, d = 11, 7, 3
        x = torch.randn(m, n, d, requires_grad=True)

        model = MLPConditioner(c, h, w, embed_dim, in_dim=d, hid_dims=hid_dims)
        in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight = model(x)

        din_proj_weight = torch.autograd.grad(in_proj_weight[0].sum(), x, create_graph=True)[0]
        din_proj_bias = torch.autograd.grad(in_proj_bias[0].sum(), x, create_graph=True)[0]
        dout_proj_weight = torch.autograd.grad(out_proj_weight[0].sum(), x, create_graph=True)[0]
        dout_proj_bias = torch.autograd.grad(out_proj_bias[0].sum(), x, create_graph=True)[0]
        dresidual_weight = torch.autograd.grad(residual_weight[0].sum(), x)[0]

        self.assertIsInstance(in_proj_weight, torch.Tensor)
        self.assertIsInstance(in_proj_bias, torch.Tensor)
        self.assertIsInstance(out_proj_weight, torch.Tensor)
        self.assertIsInstance(out_proj_bias, torch.Tensor)
        self.assertIsInstance(residual_weight, torch.Tensor)

        self.assertEqual(in_proj_weight.size(), (m, n, c, h, w, embed_dim))
        self.assertEqual(in_proj_bias.size(), (m, n, c, h, w, embed_dim))
        self.assertEqual(out_proj_weight.size(), (m, n, c, h, w, embed_dim))
        self.assertEqual(out_proj_bias.size(), (m, n, c, h, w))
        self.assertEqual(residual_weight.size(), (m, n, c, h, w))

        self.assertTrue(torch.allclose(din_proj_weight[1:], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_weight and x. ")
        self.assertTrue(torch.allclose(din_proj_bias[1:], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_bias and x. ")
        self.assertTrue(torch.allclose(dout_proj_weight[1:], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_weight and x. ")
        self.assertTrue(torch.allclose(dout_proj_bias[1:], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_bias and x. ")
        self.assertTrue(torch.allclose(dresidual_weight[1:], torch.zeros(1)),
                        msg="There's a wrong dependency between residual_weight and x. ")


class MaskedMLPConditionerTestCase(unittest.TestCase):
    def test_mlp(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        hid_dims = (36, 36)
        m, n = 11, 7
        x = torch.randn(m, n, c, h, w, requires_grad=True)

        model = MaskedMLPConditioner(c, h, w, embed_dim, hid_dims)
        in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight = model(x)

        din_proj_weight = torch.autograd.grad(in_proj_weight[..., 0, 2, 1, :].sum(), x, create_graph=True)[0]
        din_proj_bias = torch.autograd.grad(in_proj_bias[..., 1, 0, 2, :].sum(), x, create_graph=True)[0]
        dout_proj_weight = torch.autograd.grad(out_proj_weight[..., 1, 1, 3].sum(), x, create_graph=True)[0]
        dout_proj_bias = torch.autograd.grad(out_proj_bias[..., 1, 2, 1].sum(), x, create_graph=True)[0]
        dresidual_weight = torch.autograd.grad(residual_weight[..., 1, 2, 1].sum(), x)[0]

        self.assertIsInstance(in_proj_weight, torch.Tensor)
        self.assertIsInstance(in_proj_bias, torch.Tensor)
        self.assertIsInstance(out_proj_weight, torch.Tensor)
        self.assertIsInstance(out_proj_bias, torch.Tensor)
        self.assertIsInstance(residual_weight, torch.Tensor)

        self.assertEqual(in_proj_weight.size(), (m, n, c, h, w, embed_dim))
        self.assertEqual(in_proj_bias.size(), (m, n, c, h, w, embed_dim))
        self.assertEqual(out_proj_weight.size(), (m, n, c, h, w, embed_dim))
        self.assertEqual(out_proj_bias.size(), (m, n, c, h, w))
        self.assertEqual(residual_weight.size(), (m, n, c, h, w))

        self.assertTrue(torch.allclose(din_proj_weight[..., 0, 2, 1:], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_weight and x. ")
        self.assertTrue(torch.allclose(din_proj_weight[..., 1, :, :], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_weight and x. ")
        self.assertTrue(torch.allclose(din_proj_bias[..., 1, 0, 2:], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_bias and x. ")
        self.assertTrue(torch.allclose(din_proj_bias[..., 1, 1:, :], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_bias and x. ")
        self.assertTrue(torch.allclose(dout_proj_weight[..., 1, 2, 3:], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_weight and x. ")
        self.assertTrue(torch.allclose(dout_proj_weight[..., 1, 2, :], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_weight and x. ")
        self.assertTrue(torch.allclose(dout_proj_bias[..., 1, 2, 1:], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_bias and x. ")
        self.assertTrue(torch.allclose(dresidual_weight[..., 1, 2, 1:], torch.zeros(1)),
                        msg="There's a wrong dependency between residual_weight and x. ")


class LinearAttentionConditionerTestCase(unittest.TestCase):
    def test_atn(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        n = 5
        x = torch.randn(n, c, h, w, requires_grad=True)

        model = LinearAttentionConditioner(c, h, w, embed_dim)
        in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, residual_weight = model(x)

        din_proj_weight = torch.autograd.grad(in_proj_weight[..., 0, 2, 1, :].sum(), x, create_graph=True)[0]
        din_proj_bias = torch.autograd.grad(in_proj_bias[..., 1, 0, 2, :].sum(), x, create_graph=True)[0]
        dout_proj_weight = torch.autograd.grad(out_proj_weight[..., 1, 1, 3].sum(), x, create_graph=True)[0]
        dout_proj_bias = torch.autograd.grad(out_proj_bias[..., 1, 2, 1].sum(), x, create_graph=True)[0]
        dresidual_weight = torch.autograd.grad(residual_weight[..., 1, 2, 1].sum(), x)[0]

        self.assertIsInstance(in_proj_weight, torch.Tensor)
        self.assertIsInstance(in_proj_bias, torch.Tensor)
        self.assertIsInstance(out_proj_weight, torch.Tensor)
        self.assertIsInstance(out_proj_bias, torch.Tensor)
        self.assertIsInstance(residual_weight, torch.Tensor)

        self.assertEqual(in_proj_weight.size(), (n, c, h, w, embed_dim))
        self.assertEqual(in_proj_bias.size(), (n, c, h, w, embed_dim))
        self.assertEqual(out_proj_weight.size(), (n, c, h, w, embed_dim))
        self.assertEqual(out_proj_bias.size(), (n, c, h, w))
        self.assertEqual(residual_weight.size(), (n, c, h, w))

        self.assertTrue(torch.allclose(din_proj_weight[..., 0, 2, 1:], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_weight and x. ")
        self.assertTrue(torch.allclose(din_proj_weight[..., 1, :, :], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_weight and x. ")
        self.assertTrue(torch.allclose(din_proj_bias[..., 1, 0, 2:], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_bias and x. ")
        self.assertTrue(torch.allclose(din_proj_bias[..., 1, 1:, :], torch.zeros(1)),
                        msg="There's a wrong dependency between in_proj_bias and x. ")
        self.assertTrue(torch.allclose(dout_proj_weight[..., 1, 2, 3:], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_weight and x. ")
        self.assertTrue(torch.allclose(dout_proj_weight[..., 1, 2, :], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_weight and x. ")
        self.assertTrue(torch.allclose(dout_proj_bias[..., 1, 2, 1:], torch.zeros(1)),
                        msg="There's a wrong dependency between out_proj_bias and x. ")
        self.assertTrue(torch.allclose(dresidual_weight[..., 1, 2, 1:], torch.zeros(1)),
                        msg="There's a wrong dependency between residual_weight and x. ")


class SinusoidalTransformerTestCase(unittest.TestCase):
    def test_ind(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        m, n = 11, 7
        y = torch.randn(m, n, c, h, w, requires_grad=True)

        model = SinusoidalTransformer(c, h, w, embed_dim, conditioner="ind")

        # Test transform
        z, log_dets = model(y)
        dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

        self.assertIsInstance(z, torch.Tensor)
        self.assertIsInstance(log_dets, torch.Tensor)
        self.assertEqual(z.size(), y.size())
        self.assertEqual(log_dets.size(), dz_dy.size())
        self.assertEqual(log_dets.size(), (m, n))
        self.assertTrue(torch.allclose(log_dets, dz_dy))

        # Test inverse transform
        model.eval()

        model.double()
        yp = model.inv_transform(z.double(), rtol=1e-10, atol=1e-16)
        zp = model(yp)[0].float()
        model.float()

        self.assertIsInstance(yp, torch.Tensor)
        self.assertEqual(yp.size(), y.size())
        self.assertTrue(torch.allclose(zp, z))

    def test_mlp(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        m, n, d = 11, 7, 3
        hid_dims = (8, 4)

        x = torch.randn(m, n, d)
        y = torch.randn(m, n, c, h, w, requires_grad=True)

        model = SinusoidalTransformer(c, h, w, embed_dim, conditioner="mlp", in_dim=d, hid_dims=hid_dims)

        # Test transform
        z, log_dets = model(y, cond_var=x)
        dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

        self.assertIsInstance(z, torch.Tensor, f"z should be a torch.Tensor. ")
        self.assertIsInstance(log_dets, torch.Tensor, f"log_dets should be a torch.Tensor. ")
        self.assertEqual(z.size(), y.size(), f"z and y should have equal size. ")
        self.assertEqual(log_dets.size(), dz_dy.size(), f"log_dets and dz_dy should have equal size. ")
        self.assertEqual(log_dets.size(), (m, n), f"log_dets should have size {(m, n)}")
        self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4), f"log_dets and dz_dy should be equal. ")

        # Test inverse transform
        model.eval()
        yp = model.inv_transform(z, cond_var=x)
        self.assertIsInstance(yp, torch.Tensor, f"yp should be a torch.Tensor. ")
        self.assertEqual(yp.size(), y.size(), f"yp and y should have equal size. ")
        self.assertTrue(torch.allclose(yp, y, rtol=1e-4), f"yp and y should be equal. ")

    def test_masked_mlp(self):
        c, h, w = 1, 1, 5
        embed_dim = 6
        hid_dims = (36, 36)

        y = torch.randn(5, c, h, w, requires_grad=True)

        model = SinusoidalTransformer(c, h, w, embed_dim, conditioner="msk", hid_dims=hid_dims)

        # Test transform
        z, log_dets = model(y)
        jacob = []
        for i in range(z.size(-1)):
            dz_dy = torch.autograd.grad(z[:, 0, 0, i].sum(), y, create_graph=True)[0]
            jacob.append(dz_dy[:, 0, 0])
        jacob = torch.stack(jacob, dim=-1)
        dz_dy = torch.logdet(jacob)

        self.assertIsInstance(z, torch.Tensor, f"z should be a torch.Tensor. ")
        self.assertIsInstance(log_dets, torch.Tensor, f"log_dets should be a torch.Tensor. ")
        self.assertEqual(z.size(), y.size(), f"z and y should have equal size. ")
        self.assertEqual(log_dets.size(), dz_dy.size(), f"log_dets and dz_dy should have equal size. ")
        self.assertEqual(log_dets.size(), (5,), f"log_dets should have size {(5,)}")
        self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4), f"log_dets and dz_dy should be equal. ")

        # Test inverse transform
        model.eval()
        yp = model.inv_transform(z)
        self.assertIsInstance(yp, torch.Tensor, f"yp should be a torch.Tensor. ")
        self.assertEqual(yp.size(), y.size(), f"yp and y should have equal size. ")
        self.assertTrue(torch.allclose(yp, y, rtol=1e-4))

    def test_attn(self):
        c, h, w = 3, 4, 4
        embed_dim = 6

        y = torch.randn(5, c, h, w, requires_grad=True)

        model = SinusoidalTransformer(c, h, w, embed_dim, conditioner="atn")

        # Test transform
        z, log_dets = model(y)

        def f(x):
            return model(x)[0]

        jacobs = []
        jacob = jacobian(f, y)
        for b in range(y.size(0)):
            jacobs.append(jacob[b, :, :, :, b, :, :, :].reshape(c * h * w, c * h * w))

        jacobs = torch.stack(jacobs, dim=0)
        dz_dy = torch.logdet(jacobs)

        self.assertIsInstance(z, torch.Tensor, f"z should be a torch.Tensor. ")
        self.assertIsInstance(log_dets, torch.Tensor, f"log_dets should be a torch.Tensor. ")
        self.assertEqual(z.size(), y.size(), f"z and y should have equal size. ")
        self.assertEqual(log_dets.size(), dz_dy.size(), f"log_dets and dz_dy should have equal size. ")
        self.assertEqual(log_dets.size(), (5,), f"log_dets should have size {(5,)}")
        self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4), f"log_dets and dz_dy should be equal. ")

        # Test inverse transform
        model.eval()
        yp = model.inv_transform(z)
        self.assertIsInstance(yp, torch.Tensor, f"yp should be a torch.Tensor. ")
        self.assertEqual(yp.size(), y.size(), f"yp and y should have equal size. ")
        self.assertTrue(torch.allclose(yp, y, rtol=1e-4))


class SinusoidalFlowTestCase(unittest.TestCase):
    def test_ind(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        m, n = 11, 7
        num_layers = 12

        for affine in [True, False]:
            y = torch.randn(m, n, c, h, w, requires_grad=True)

            model = SinusoidalFlow(c, h, w, embed_dim, conditioner="ind", num_layers=num_layers, affine=affine)

            # Test transform
            z, log_joint_dens = model(y)
            log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))
            dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor,
                                  f"When affine = {affine}, z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When affine = {affine}, log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When affine = {affine}, z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When affine = {affine}, log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (m, n),
                             f"When affine = {affine}, log_dets should have size {(m, n)}")
            self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-3),
                            f"When affine = {affine}, log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()
            model.double()
            yp = model.inv_transform(z.double(), rtol=1e-8, atol=1e-14).float()

            self.assertIsInstance(yp, torch.Tensor,
                                  f"When affine = {affine}, yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When affine = {affine}, yp and y should have equal size. ")
            self.assertTrue(torch.allclose(yp, y, rtol=1e-5),
                            f"When and affine = {affine}, yp and y should be equal. ")

    def test_mlp(self):
        c, h, w = 2, 3, 4
        embed_dim = 6
        m, n, d = 11, 7, 3
        hid_dims = (8, 4)
        num_layers = 3

        for affine in [True, False]:
            x = torch.randn(m, n, d)
            y = torch.randn(m, n, c, h, w, requires_grad=True)

            model = SinusoidalFlow(c, h, w, embed_dim, conditioner="ind", num_layers=num_layers,
                                   affine=affine, in_dim=d, hid_dims=hid_dims)

            # Test transform
            z, log_joint_dens = model(y, cond_var=x)
            log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))
            dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor,
                                  f"When affine = {affine}, z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When affine = {affine}, log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When affine = {affine}, z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When affine = {affine}, log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (m, n),
                             f"When affine = {affine}, log_dets should have size {(m, n)}")
            self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-5),
                            f"When affine = {affine}, log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()
            yp = model.inv_transform(z, cond_var=x)
            self.assertIsInstance(yp, torch.Tensor,
                                  f"When affine = {affine}, yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When affine = {affine}, yp and y should have equal size. ")
            self.assertTrue(torch.allclose(yp, y, rtol=1e-5),
                            f"When affine = {affine}, yp and y should be equal. ")


class DepthwiseSinusoidalTransformerTestCase(unittest.TestCase):

    def test_masked_mlp(self):
        c, h, w = 3, 4, 4
        embed_dim = 6
        hid_dims = (300,)

        y = torch.randn(5, c, h, w, requires_grad=True)

        model = DepthwiseSinusoidalTransformer(c, h, w, embed_dim, conditioner="msk", hid_dims=hid_dims)

        # Test transform
        z, log_dets = model(y)

        def f(x):
            return model(x)[0]

        jacobs = []
        jacob = jacobian(f, y)
        for b in range(y.size(0)):
            jacobs.append(jacob[b, :, :, :, b, :, :, :].reshape(c * h * w, c * h * w))

        jacobs = torch.stack(jacobs, dim=0)
        dz_dy = torch.logdet(jacobs)

        self.assertIsInstance(z, torch.Tensor, f"z should be a torch.Tensor. ")
        self.assertIsInstance(log_dets, torch.Tensor, f"log_dets should be a torch.Tensor. ")
        self.assertEqual(z.size(), y.size(), f"z and y should have equal size. ")
        self.assertEqual(log_dets.size(), dz_dy.size(), f"log_dets and dz_dy should have equal size. ")
        self.assertEqual(log_dets.size(), (5,), f"log_dets should have size {(5,)}")
        self.assertTrue(torch.allclose(log_dets, dz_dy, rtol=1e-4), f"log_dets and dz_dy should be equal. ")

        # Test inverse transform
        model.eval()
        yp = model.inv_transform(z)
        self.assertIsInstance(yp, torch.Tensor, f"yp should be a torch.Tensor. ")
        self.assertEqual(yp.size(), y.size(), f"yp and y should have equal size. ")
        self.assertTrue(torch.allclose(yp, y, rtol=1e-4))
