import unittest
import torch

from project.models.affine import IndependentConditioner, MLPConditioner, AffineTransformer, AffineFlow, \
    AffineFlowRegressor


class IndependentConditionerTestCase(unittest.TestCase):

    def test_output(self):
        c, h, w = 1, 1, 1
        m, n, d = 11, 7, 3
        x = torch.randn(m, n, d, requires_grad=True)
        model = IndependentConditioner(c, h, w)

        log_std, mean = model(x)
        dlog_std = torch.autograd.grad(log_std.sum(), x, create_graph=True, allow_unused=True)[0]
        dmean = torch.autograd.grad(mean.sum(), x, allow_unused=True)[0]

        self.assertIsInstance(log_std, torch.nn.Parameter)
        self.assertIsInstance(mean, torch.nn.Parameter)
        self.assertEqual(log_std.size(), (c, h, w))
        self.assertEqual(mean.size(), (c, h, w))
        self.assertIsNone(dlog_std, msg="log_std shouldn't depend on x. ")
        self.assertIsNone(dmean, msg="mean shouldn't depend on x. ")


class MLPConditionerTestCase(unittest.TestCase):
    def setUp(self):
        self.c, self.h, self.w = 2, 3, 4
        self.hid_dims = (8, 4)
        self.m, self.n, self.d = 11, 7, 3
        self.x = torch.randn(self.m, self.n, self.d, requires_grad=True)

    def test_use_ind_std(self):
        model = MLPConditioner(self.c, self.h, self.w, self.d, self.hid_dims, use_ind_std=True)
        log_std, mean = model(self.x)
        dlog_std = torch.autograd.grad(log_std.sum(), self.x, create_graph=True, allow_unused=True)[0]
        dmean = torch.autograd.grad(mean[0].sum(), self.x)[0]

        self.assertIsInstance(log_std, torch.nn.Parameter)
        self.assertIsInstance(mean, torch.Tensor)
        self.assertEqual(log_std.size(), (self.c, self.h, self.w))
        self.assertEqual(mean.size(), (self.m, self.n, self.c, self.h, self.w))
        self.assertIsNone(dlog_std, msg="When use_ind_std=True, log_std shouldn't depend on x. ")
        self.assertTrue(torch.isclose(dmean[1:], torch.zeros(1)).all(),
                        msg="There's a wrong dependency between mean and x. ")

    def test_not_use_ind_std(self):
        model = MLPConditioner(self.c, self.h, self.w, self.d, self.hid_dims, use_ind_std=False)
        log_std, mean = model(self.x)
        dlog_std = torch.autograd.grad(log_std[0].sum(), self.x, create_graph=True)[0]
        dmean = torch.autograd.grad(mean[0].sum(), self.x)[0]

        self.assertIsInstance(log_std, torch.Tensor)
        self.assertIsInstance(mean, torch.Tensor)
        self.assertEqual(log_std.size(), (self.m, self.n, self.c, self.h, self.w))
        self.assertEqual(mean.size(), (self.m, self.n, self.c, self.h, self.w))
        self.assertTrue(torch.isclose(dlog_std[1:], torch.zeros(1)).all(),
                        msg="There's a wrong dependency between log_std and x. ")
        self.assertTrue(torch.isclose(dmean[1:], torch.zeros(1)).all(),
                        msg="There's a wrong dependency between mean and x. ")


class AffineTransformerTestCase(unittest.TestCase):

    def test_ind(self):
        c, h, w = 2, 3, 4
        m, n = 11, 7
        y = torch.randn(m, n, c, h, w, requires_grad=True)

        model = AffineTransformer(c, h, w, conditioner="ind")

        # Test transform
        z, log_dets = model(y)
        dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

        self.assertIsInstance(z, torch.Tensor)
        self.assertIsInstance(log_dets, torch.Tensor)
        self.assertEqual(z.size(), y.size())
        self.assertEqual(log_dets.size(), dz_dy.size())
        self.assertEqual(log_dets.size(), (m, n))
        self.assertTrue(torch.isclose(log_dets, dz_dy).all())

        # Test inverse transform
        model.eval()
        yp = model.inv_transform(z)
        self.assertIsInstance(yp, torch.Tensor)
        self.assertEqual(yp.size(), y.size())
        self.assertTrue(torch.isclose(yp, y).all())

    def test_mlp(self):
        c, h, w = 2, 3, 4
        m, n, d = 11, 7, 3
        hid_dims = (8, 4)

        for use_ind_std in [True, False]:
            x = torch.randn(m, n, d)
            y = torch.randn(m, n, c, h, w, requires_grad=True)

            model = AffineTransformer(c, h, w, conditioner="mlp", in_dim=d, hid_dims=hid_dims, use_ind_std=use_ind_std)

            # Test transform
            z, log_dets = model(y, cond_var=x)
            dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor,
                                  f"When use_ind_std = {use_ind_std}, z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When use_ind_std = {use_ind_std}, log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When use_ind_std = {use_ind_std}, z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When use_ind_std = {use_ind_std}, log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (m, n),
                             f"When use_ind_std = {use_ind_std}, log_dets should have size {(m, n)}")
            self.assertTrue(torch.isclose(log_dets, dz_dy).all(),
                            f"When use_ind_std = {use_ind_std}, log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()
            yp = model.inv_transform(z, cond_var=x)
            self.assertIsInstance(yp, torch.Tensor,
                                  f"When use_ind_std = {use_ind_std}, yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When use_ind_std = {use_ind_std}, yp and y should have equal size. ")
            self.assertTrue(torch.isclose(yp, y, atol=1e-7).all(),
                            f"When use_ind_std = {use_ind_std}, yp and y should be equal. ")


class AffineFlowTestCase(unittest.TestCase):
    def test_ind(self):
        c, h, w = 2, 3, 4
        m, n = 11, 7
        num_layers = 3

        for use_norm in [True, False]:
            y = torch.randn(m, n, c, h, w, requires_grad=True)

            model = AffineFlow(c, h, w, conditioner="ind", num_layers=num_layers, use_norm=use_norm)

            # Test transform
            z, log_joint_dens = model(y)
            log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))
            dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

            self.assertIsInstance(z, torch.Tensor,
                                  f"When use_norm = {use_norm}, z should be a torch.Tensor. ")
            self.assertIsInstance(log_dets, torch.Tensor,
                                  f"When use_norm = {use_norm}, log_dets should be a torch.Tensor. ")
            self.assertEqual(z.size(), y.size(),
                             f"When use_norm = {use_norm}, z and y should have equal size. ")
            self.assertEqual(log_dets.size(), dz_dy.size(),
                             f"When use_norm = {use_norm}, log_dets and dz_dy should have equal size. ")
            self.assertEqual(log_dets.size(), (m, n),
                             f"When use_norm = {use_norm}, log_dets should have size {(m, n)}")
            self.assertTrue(torch.isclose(log_dets, dz_dy, atol=1e-5).all(),
                            f"When use_norm = {use_norm}, log_dets and dz_dy should be equal. ")

            # Test inverse transform
            model.eval()
            y = torch.randn(m, n, c, h, w)  # use a fresh y to test the normalisation layers
            with torch.no_grad():
                z, _ = model(y)
            yp = model.inv_transform(z)

            self.assertIsInstance(yp, torch.Tensor,
                                  f"When use_norm = {use_norm}, yp should be a torch.Tensor. ")
            self.assertEqual(yp.size(), y.size(),
                             f"When use_norm = {use_norm}, yp and y should have equal size. ")
            self.assertTrue(torch.isclose(yp, y, atol=1e-5).all(),
                            f"When and use_norm = {use_norm}, yp and y should be equal. ")

    def test_mlp(self):
        c, h, w = 2, 3, 4
        m, n, d = 11, 7, 3
        num_layers = 3
        hid_dims = (8, 4)
        k = 5

        for use_ind_std in [True, False]:
            for use_norm in [True, False]:
                x = torch.randn(m, n, d)
                y = torch.randn(k, m, n, c, h, w, requires_grad=True)

                model = AffineFlow(c, h, w, conditioner="mlp", num_layers=num_layers, use_norm=use_norm,
                                   in_dim=d, hid_dims=hid_dims, use_ind_std=use_ind_std)

                # Test transform
                z, log_joint_dens = model(y, cond_var=x)
                log_dets = log_joint_dens - torch.sum(model.base_dist.log_prob(z), dim=(-3, -2, -1))
                dz_dy = torch.autograd.grad(z.sum(), y)[0].log().sum(dim=(-3, -2, -1))

                self.assertIsInstance(z, torch.Tensor,
                                      f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, z should be a torch.Tensor. ")
                self.assertIsInstance(log_dets, torch.Tensor,
                                      f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, log_dets should be a torch.Tensor. ")
                self.assertEqual(z.size(), y.size(),
                                 f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, z and y should have equal size. ")
                self.assertEqual(log_dets.size(), dz_dy.size(),
                                 f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, log_dets and dz_dy should have equal size. ")
                self.assertEqual(log_dets.size(), (k, m, n),
                                 f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, log_dets should have size {(k, m, n)}")
                self.assertTrue(torch.isclose(log_dets, dz_dy, atol=1e-5).all(),
                                f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, log_dets and dz_dy should be equal. ")

                # Test inverse transform
                model.eval()
                x = torch.randn(m, n, d)
                y = torch.randn(k, m, n, c, h, w)  # use a fresh y to test the normalisation layers
                with torch.no_grad():
                    z, _ = model(y, cond_var=x)
                yp = model.inv_transform(z, cond_var=x)

                self.assertIsInstance(yp, torch.Tensor,
                                      f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, yp should be a torch.Tensor. ")
                self.assertEqual(yp.size(), y.size(),
                                 f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, yp and y should have equal size. ")
                self.assertTrue(torch.isclose(yp, y, atol=1e-5).all(),
                                f"When use_ind_std = {use_ind_std} and use_norm = {use_norm}, yp and y should be equal. ")


class AffineFlowRegressorTestCase(unittest.TestCase):
    def test_calc_modes(self):
        n, d = 7, 3
        num_layers = 3
        hid_dims = (8, 4)

        x = torch.randn(n, d)
        model = AffineFlowRegressor(conditioner="mlp", num_layers=num_layers, use_norm=True,
                                    in_dim=d, hid_dims=hid_dims, use_ind_std=False)

        model.eval()
        means = model.predict(x).reshape(-1, 1)
        modes, _ = list(zip(*model.calc_modes(x)))
        modes = torch.stack(modes, dim=0)

        self.assertIsInstance(means, torch.Tensor)
        self.assertEqual(means.size(), modes.size())
        self.assertTrue(torch.allclose(means, modes))
        # for i, y in enumerate(torch.unbind(means, dim=0)):
        #     self.assertTrue(torch.isclose(y.flatten(), modes[i]).all())

    def test_calc_mean(self):
        n, d = 7, 3
        num_layers = 3
        hid_dims = (8, 4)

        x = torch.randn(n, d)
        model = AffineFlowRegressor(conditioner="mlp", num_layers=num_layers, use_norm=True,
                                    in_dim=d, hid_dims=hid_dims, use_ind_std=False)

        model.eval()
        means = model.predict(x).reshape(-1, 1)
        pred_means = model.calc_mean(x)

        self.assertIsInstance(means, torch.Tensor)
        self.assertIsInstance(pred_means, torch.Tensor)
        self.assertEqual(means.size(), pred_means.size())
        self.assertTrue(torch.allclose(means, pred_means))
