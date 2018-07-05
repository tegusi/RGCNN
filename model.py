import torch
import torch.nn as nn
import torch.nn.functional as F

class GetGraph(nn.Module):
    def __init__(self):
        super(GetGraph, self).__init__()

    def forward(self, point_cloud):
        point_cloud_transpose = point_cloud.permute(0, 2, 1)
        point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = -2 * point_cloud_inner
        point_cloud_square = torch.sum(torch.mul(point_cloud, point_cloud), dim=2, keepdim=True)
        point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
        adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
        adj_matrix = torch.exp(-adj_matrix)
        return adj_matrix


class GetLaplacian(nn.Module):
    def __init__(self, normalize=True):
        super(GetLaplacian, self).__init__()
        self.normalize = normalize

    def diag(self, mat):
        # input is batch x vertices
        d = []
        for vec in mat:
            d.append(torch.diag(vec))
        return torch.stack(d)

    def forward(self, adj_matrix):
        if self.normalize:
            D = torch.sum(adj_matrix, dim=2)
            eye = torch.ones_like(D)
            eye = self.diag(eye)
            D = 1 / torch.sqrt(D)
            D = self.diag(D)
            L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
        else:
            D = torch.sum(adj_matrix, dim=1)
            D = torch.matrix_diag(D)
            L = D - adj_matrix
        return L


class GetFilter(nn.Module):
    def __init__(self, Fin, K, Fout):
        super(GetFilter, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.K = K
        self.W = nn.Parameter(torch.Tensor(self.K * self.Fin, self.Fout))
        nn.init.normal_(self.W, mean=0, std=0.2)
        self.B = nn.Parameter(torch.Tensor(self.Fout))
        nn.init.normal_(self.B, mean=0, std=0.2)
        self.relu = nn.ReLU()

    # def reset_parameters(self):

    def forward(self, x, L):
        N, M, Fin = list(x.size())
        K = self.K
        x0 = x.clone()
        x = x0.unsqueeze(0)

        #         x = x.expand(-1,-1,-1,1)
        def concat(x, x_):
            x_ = x_.unsqueeze(0)
            #             x_ = x.expand(1,-1,-1)
            return torch.cat((x, x_), dim=0)

        if K > 1:
            x1 = torch.matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * torch.matmul(L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = x.permute(1, 2, 3, 0)
        x = x.reshape(N * M, Fin * K)
        x = torch.matmul(x, self.W)
        x = torch.add(x, self.B)
        x = self.relu(x)
        return x.reshape(N, M, self.Fout)


class RGCNN_Seg(nn.Module):
    def __init__(self, vertice, F, K, M, regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name=''):

        # Verify the consistency w.r.t. the number of layers.
        assert len(F) == len(K)

        super(RGCNN_Seg, self).__init__()
        # Keep the useful Laplacians only. May be zero.
        self.vertice = vertice
        # Print information about NN architecture.
        # Ngconv = len(F)
        # Nfc = len(M)
        # print('NN architecture')
        # print('  input: M_0 = {}'.format(vertice))
        # for i in range(Ngconv):
        #     print('  layer {0}: gconv{0}'.format(i + 1))
        #     print('    representation: M_{0} * F_{1}= {2} * {3} = {4}'.format(
        #         i, i + 1, vertice, F[i], vertice * F[i]))
        #     F_last = F[i - 1] if i > 0 else 1
        #     print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
        #         i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
        #     if brelu == 'b1relu':
        #         print('    biases: F_{} = {}'.format(i + 1, F[i]))
        #     elif brelu == 'b2relu':
        #         print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
        #             i + 1, vertice, F[i], vertice * F[i]))
        # for i in range(Nfc):
        #     name = 'fc{}'.format(i + 1)
        #     print('  layer {}: {}'.format(Ngconv + i + 1, name))
        #     print('    representation: M_{} = {}'.format(Ngconv + i + 1, M[i]))
        #     M_last = M[i - 1] if i > 0 else M_0 if Ngconv == 0 else vertice * F[-1]
        #     print('    weights: M_{} * M_{} = {} * {} = {}'.format(
        #         Ngconv + i, Ngconv + i + 1, M_last, M[i], M_last * M[i]))
        #     print('    biases: M_{} = {}'.format(Ngconv + i + 1, M[i]))

        # Operations
        self.getGraph = GetGraph()
        self.getLaplacian = GetLaplacian(normalize=True)
        # Store attributes and bind operations.
        self.F, self.K, self.M = F, K, M
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        for i in range(len(F)):
            if i == 0:
                layer = GetFilter(Fin=6, K=K[i], Fout=F[i])
            else:
                layer = GetFilter(Fin=F[i - 1], K=K[i], Fout=F[i])
            setattr(self, 'gcn%d' % i, layer)

    def forward(self, x, cat):
        L = self.getGraph(x)
        L = self.getLaplacian(L)
        #         cat = torch.unsqueeze(cat,1)
        #         cat = torch.zeros(self.batch_size, self.class_size).scatter_(1, cat, 1)
        #         cat = torch.unsqueeze(cat,1)
        #         cat = cat.expand(-1,self.vertice,-1).double()
        #         x = torch.cat((x,cat),dim=2)
        for i in range(len(self.F)):
            x = getattr(self, 'gcn%d' % i)(x, L)
        return x

class RGCNN_Cls(nn.Module):
    def __init__(self, vertice, F, K, M, regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name=''):

        # Verify the consistency w.r.t. the number of layers.
        assert len(F) == len(K)

        super(RGCNN_Cls, self).__init__()
        # Keep the useful Laplacians only. May be zero.
        self.vertice = vertice
        # Print information about NN architecture.
        Ngconv = len(F)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(vertice))
        for i in range(Ngconv):
            print('  layer {0}: gconv{0}'.format(i + 1))
            print('    representation: M_{0} * F_{1}= {2} * {3} = {4}'.format(
                i, i + 1, vertice, F[i], vertice * F[i]))
            F_last = F[i - 1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
            print('    biases: F_{} = {}'.format(i + 1, F[i]))
        for i in range(Nfc):
            name = 'fc{}'.format(i + 1)
            print('  layer {}: {}'.format(Ngconv + i + 1, name))
            print('    representation: M_{} = {}'.format(Ngconv + i + 1, M[i]))
            M_last = M[i - 1] if i > 0 else vertice if Ngconv == 0 else vertice * F[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv + i, Ngconv + i + 1, M_last, M[i], M_last * M[i]))
            print('    biases: M_{} = {}'.format(Ngconv + i + 1, M[i]))

        # Operations
        self.getGraph = GetGraph()
        self.getLaplacian = GetLaplacian(normalize=True)
        # Store attributes and bind operations.
        self.F, self.K, self.M = F, K, M
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.pool = nn.MaxPool1d(vertice)
        self.relu = nn.ReLU()
        for i in range(len(F)):
            if i == 0:
                layer = GetFilter(Fin=6, K=K[i], Fout=F[i])
            else:
                layer = GetFilter(Fin=F[i - 1], K=K[i], Fout=F[i])
            setattr(self, 'gcn%d' % i, layer)
        for i in range(len(M)):
            if i == 0:
                layer = nn.Linear(F[-1], M[i])
            else:
                layer = nn.Linear(M[i-1], M[i])
            setattr(self, 'fc%d' % i, layer)

    def forward(self, x, cat):
        L = self.getGraph(x)
        L = self.getLaplacian(L)
        #         cat = torch.unsqueeze(cat,1)
        #         cat = torch.zeros(self.batch_size, self.class_size).scatter_(1, cat, 1)
        #         cat = torch.unsqueeze(cat,1)
        #         cat = cat.expand(-1,self.vertice,-1).double()
        #         x = torch.cat((x,cat),dim=2)
        for i in range(len(self.F)):
            x = getattr(self, 'gcn%d' % i)(x, L)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x.squeeze_(2)
        for i in range(len(self.M)):
            x = getattr(self, 'fc%d' % i)(x)
            # x = self.relu(x)
        return x
