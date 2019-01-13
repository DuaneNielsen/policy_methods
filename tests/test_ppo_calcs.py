import torch
import torch.nn as nn
from unittest import TestCase
import matplotlib.pyplot as plt


class PPOTests(TestCase):

    def calc(self, action, newprob, oldprob, advantage):

        action = torch.tensor([action])
        advantage = torch.tensor([advantage])

        newprob_t = torch.tensor([newprob], requires_grad=True)
        oldprob_t = torch.tensor([oldprob])

        clip = 0.999

        newprob = action * (newprob_t + 1e-12) + (1.0 - action) * (1.0 - newprob_t + 1e-12)
        oldprob = action * (oldprob_t + 1e-12) + (1.0 - action) * (1.0 - oldprob_t + 1e-12)


        # compute the surrogate
        ratio = (newprob) / (oldprob)

        clipped_step = ratio.clamp(min=0.8, max=1.2) * advantage
        full_step = ratio * advantage

        min_step = torch.stack((full_step, clipped_step), dim=1)
        min_step, clipped = torch.min(min_step, dim=1)

        min_step.mean().backward()

        # min_step *= -1.0

        print(f'ACT__ {action[0].data}')
        print(f'ADVTG {advantage[0].data}')
        print(f'CHNGE {(newprob[0] - oldprob[0]).data}')
        print(f'NEW__ {newprob_t[0].data}')
        print(f'OLD__ {oldprob_t[0].data}')
        print(f'NEW_P {newprob[0].data}')
        print(f'OLD_P {oldprob[0].data}')
        print(f'RATIO {ratio[0].data}')
        print(f'CLIP_ {clipped_step[0].data}')
        print(f'NEW_G {newprob_t.grad.data.item()}')

        return action, advantage, oldprob_t, newprob_t, ratio

    def test_ppo(self):

        # unclipped
        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=1.0, advantage=1.0, newprob=0.8, oldprob=0.7)
        print('')
        assert newprob.grad > 0

        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=1.0, advantage=-1.0, newprob=0.8, oldprob=0.7)
        print('')
        assert newprob.grad < 0

        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=0.0, advantage=1.0, newprob=0.3, oldprob=0.2)
        print('')
        assert newprob.grad < 0

        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=0.0, advantage=-1.0, newprob=0.3, oldprob=0.2)
        print('')
        assert newprob.grad > 0


        # clipped
        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=1.0, advantage=1.0, newprob=0.2, oldprob=0.1)
        print('')
        assert newprob.grad == 0

        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=1.0, advantage=-1.0, newprob=0.2, oldprob=0.1)
        print('')
        assert newprob.grad == 0

        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=0.0, advantage=1.0, newprob=0.8, oldprob=0.7)
        print('')
        assert newprob.grad == 0

        action, advantage, oldprob, newprob, ratio = \
        self.calc(action=0.0, advantage=-1.0, newprob=0.8, oldprob=0.7)
        print('')
        assert newprob.grad == 0

    def test_plot_ratio(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(0.0, 1.0, 0.05)
        Y = np.arange(0.0, 1.0, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = X / Y

        # Plot the surface.
        surf = ax.scatter(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('ratio')

        plt.show()


    def test_clipped_plot_ratio(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        oldprob = torch.linspace(0.0, 1.0, 20)
        newprob = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob, newprob))
        ratio = newprob / oldprob
        clipped = ratio.clamp(min=0.8, max=1.2) * 1.0

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.numpy(), clipped.numpy(), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('clipped ratio')

        plt.show()


    def test_clipped_plot_ratio_neg_advantage(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        oldprob = torch.linspace(0.0, 1.0, 20)
        newprob = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob, newprob))

        ratio = newprob / oldprob

        clipped = ratio.clamp(min=0.8, max=1.2) * -1.0

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.numpy(), clipped.numpy(), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('clipped ratio with negative advantage')

        plt.show()

    def test_clipped_plot_grad(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        oldprob_t = torch.linspace(0.0, 1.0, 20)
        newprob_t = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob_t, newprob_t))
        newprob.retain_grad()
        newprob.requires_grad = True
        ratio = newprob / oldprob
        clipped = ratio.clamp(min=0.8, max=1.2) * 1.0
        clipped.mean().backward()


        grad = newprob.grad.data.numpy()

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.detach().numpy(), grad, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('clipped gradient of newprob')

        plt.show()

    def test_clipped_negadvantage_plot_grad(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        oldprob_t = torch.linspace(0.0, 1.0, 20)
        newprob_t = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob_t + 1e-12, newprob_t))
        newprob.retain_grad()
        newprob.requires_grad = True
        ratio = newprob / oldprob
        clipped = ratio.clamp(min=0.8, max=1.2) * -1.0
        clipped.mean().backward()


        grad = newprob.grad.data.numpy()

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.detach().numpy(), grad, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('clipped gradient of newprob')

        plt.show()

    def test_min_clipped_plot_grad(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        advantage = 1.0

        # Make data.
        oldprob_t = torch.linspace(0.0, 1.0, 20)
        newprob_t = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob_t, newprob_t))
        newprob.retain_grad()
        newprob.requires_grad = True
        ratio = newprob / oldprob
        full = ratio * advantage
        clipped = ratio.clamp(min=0.8, max=1.2) * advantage

        min_step = torch.stack((full, clipped), dim=1)
        min_step, clipped = torch.min(min_step, dim=1)

        min_step.sum().backward()

        grad = newprob.grad.data.numpy()

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.detach().numpy(), grad, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title('gradients of min clipped, positive advantage')
        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('clipped gradient of newprob')

        plt.show()

    def test_min_clipped_negadvantage_plot_grad(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        advantage = -1.0

        # Make data.
        oldprob_t = torch.linspace(0.0, 1.0, 20)
        newprob_t = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob_t, newprob_t))
        newprob.retain_grad()
        newprob.requires_grad = True
        ratio = newprob / oldprob
        full = ratio * advantage
        clipped = ratio.clamp(min=0.8, max=1.2) * advantage

        min_step = torch.stack((full, clipped), dim=1)
        min_step, clipped = torch.min(min_step, dim=1)

        min_step.sum().backward()

        grad = newprob.grad.data.numpy()

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.detach().numpy(), grad, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title('gradients of min clipped, negative advantage')
        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('clipped gradient of newprob')

        plt.show()

    def test_min_clipped_negadvantage_plot_grad(self):
        '''
        ======================
        3D surface (color map)
        ======================

        Demonstrates plotting a 3D surface colored with the coolwarm color map.
        The surface is made opaque by using antialiased=False.

        Also demonstrates using the LinearLocator and custom formatting for the
        z axis tick labels.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        advantage = -1.0

        # Make data.
        oldprob_t = torch.linspace(0.0, 1.0, 20)
        newprob_t = torch.linspace(0.0, 1.0, 20)
        oldprob, newprob = torch.meshgrid((oldprob_t, newprob_t))
        newprob.retain_grad()
        newprob.requires_grad = True
        ratio = newprob / oldprob

        ratio.sum().backward()

        grad = newprob.grad.data.numpy()

        # Plot the surface.
        surf = ax.scatter(oldprob.numpy(), newprob.detach().numpy(), grad, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title('gradients of ratio, unclipped')
        plt.xlabel('new probs')
        plt.ylabel('old probs')
        ax.set_zlabel('gradient of newprob')

        plt.show()