import numpy as np
import pickle
import torch
from torch.nn import Module
from torch.autograd import Variable
import os

torch_version = float( torch.__version__[0:3] )

# -----------------------------------------------------------------------------

class SMPLModel(Module):
    def __init__(self, device=None, model_path=None):
        super(SMPLModel, self).__init__()
        # with open(model_path, 'rb') as f:
        #   params = pickle.load(f)
        with open(model_path, 'rb') as f:
            params = np.load( f )
            print( "loaded smpl model from %s" % ( model_path ) )

        self.J_regressor = torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(torch.float64)
        self.weights = torch.from_numpy(params['weights']).type(torch.float64)
        self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64)
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float64)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float64)
        self.kintree_table = params['kintree_table']
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        for name in ['J_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        #r = r.to(self.device)
        eps = r.clone().normal_(std=1e-8)

        if torch_version >= 1.0:
            theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        else:
            theta = torch.norm( (r+eps).squeeze(1), dim=1 )[:,None,None]
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
                         + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64).to(x.device)
        ret = torch.cat((x, ones), dim=0)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros((x.shape[0], 4, 3), dtype=torch.float64).to(x.device)
        ret = torch.cat((zeros43, x), dim=2)
        return ret

    def write_obj(self, verts, file_name):
        write_simple_obj( verts, self.faces, file_name )
    
    def write_change_obj(self, verts, file_name):
        combined_v = torch.cat( (verts, self.v_template), dim=0 )
        combined_f = np.concatenate( (self.faces, self.faces+verts.size(0)), axis=0 )
        write_simple_obj( combined_v, combined_f, file_name )

    def forward(self, betas, pose, trans, simplify=False):
        """
        Construct a compute graph that takes in parameters and outputs a tensor as
        model vertices. Face indices are also returned as a numpy ndarray.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        betas: Parameter for model shape. A tensor of shape [10] as coefficients of
        PCA components. Only 10 components were released by SMPL author.

        trans: Global translation tensor of shape [3].

        Return:
        ------
        A tensor for vertices, and a numpy ndarray as face indices.

        """
        id_to_col = {self.kintree_table[1, i]: i
                                 for i in range(self.kintree_table.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        if torch_version >= 1.0:
            v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
        else:
            v_shaped = torch.matmul(self.shapedirs, betas) + self.v_template

        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) + \
                torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float64)).to(self.device)
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            if torch_version >= 1.0:
                v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))
            else:
                v_posed = v_shaped + torch.matmul(self.posedirs, lrotmin)

        results = []
        results.append(
            self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
        )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (R_cube_big[i], torch.reshape(J[i, :] - J[parent[i], :], (3, 1))),
                            dim=1
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=0)
        results = stacked - \
            self.pack(
                torch.matmul(
                    stacked,
                    torch.reshape(
                        torch.cat((J, torch.zeros((24, 1), dtype=torch.float64).to(self.device)), dim=1),
                        (24, 4, 1)
                    )
                )
            )
        if torch_version >= 1.0:
            T = torch.tensordot(self.weights, results, dims=([1], [0]))
        else:
            T = torch.matmul(self.weights, results.view(-1,4*4)).view(-1,4,4)
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((v_posed.shape[0], 1), dtype=torch.float64).to(self.device)), dim=1
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        result = v + torch.reshape(trans, (1, 3))
        return result

# -----------------------------------------------------------------------------

def write_simple_obj(verts, faces, file_name, vert_color=None):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        if vert_color is not None:
            for vc in vert_color:
                fp.write('vc %d %d %d\n' % (vc[0], vc[1], vc[2]))

# -----------------------------------------------------------------------------

def get_device(gpu_id):
    if len(gpu_id) > 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    return device

# -----------------------------------------------------------------------------

def test_gpu(gpu_id=[0]):

    device = get_device(gpu_id)

    # parms
    pose_size = 72
    beta_size = 10
    np.random.seed(9608)
    pose = torch.from_numpy((np.random.rand(pose_size) - 0.5) * 0.4).type(torch.float64).to(device)
    betas = torch.from_numpy((np.random.rand(beta_size) - 0.5) * 0.06).type(torch.float64).to(device)
    trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)
    pose  = Variable( pose, requires_grad=True )
    betas = Variable( betas, requires_grad=True )
    trans = Variable( trans, requires_grad=True )
    # pose.zero_()
    # betas.zero_()
    # trans.zero_()

    # forward
    model_path = './models/model_female_py3.pkl'
    model = SMPLModel(device=device, model_path=model_path)
    result = model(betas, pose, trans)

    # save
    outmesh_path = './test/smpl_torch.obj'
    model.write_obj(result, outmesh_path)

    # backward
    fake_loss = result.norm()
    fake_loss.backward()
    import ipdb; ipdb.set_trace()

# -----------------------------------------------------------------------------

def test_pose(gpu_id=[0]):

    device = get_device(gpu_id)

    # parms
    pose_size = 72
    beta_size = 10
    np.random.seed(9608)
    pose = torch.from_numpy(np.zeros(pose_size)).type(torch.float64).to(device)
    betas = torch.from_numpy(np.zeros(beta_size)).type(torch.float64).to(device)
    trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)

    # forward
    model_path = './models/model_female_py3.pkl'
    model = SMPLModel(device=device, model_path=model_path)

    for pose_id in range(72):
        joint_id = pose_id / 3
        axis_id = pose_id % 3
        this_pose = pose.clone()
        this_pose[ pose_id ] = 0.6
        result = model(betas, this_pose, trans)
        outmesh_path = './test/poses/smpl_joint_%02d_axis_%d.obj' % ( joint_id, axis_id )
        model.write_obj(result, outmesh_path)

    import ipdb; ipdb.set_trace()

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    test_gpu([1])

    # test_pose([1])
